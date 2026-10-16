#!/usr/bin/env python3
"""
lidar_sim.py - standalone, physics-based simulation of a spinning pulsed LiDAR
==============================================================================

Only dependency: numpy (matplotlib = optional static PNG, plotly = optional interactive HTML).

What it models (per laser pulse)
--------------------------------
1. Scan geometry     N channels x azimuth steps, rotation rate, per-pulse firing time,
                     optional sensor motion during the sweep (rolling-scan distortion).
2. Beam              Gaussian beam of finite divergence, sampled by K sub-rays whose
                     weights are the Gaussian beam profile (so edges give "mixed pixels").
3. Target physics    Ray/primitive intersection, surface normal, incidence angle, material
                     BRDF = Lambertian + optional specular lobe.
4. LiDAR equation    P_r = P_t * A_r * eta * f_r(theta) * cos(theta) / R^2 * exp(-2*alpha*R)
                     with f_r = rho_d/pi + rho_s*(n+2)/(2*pi)*cos(2*theta)^n   (monostatic).
                     For a Lambertian target that fills the beam this is the classic
                     P_r = P_t * rho * cos(theta) * A_r / (pi R^2) * T_atm^2.
5. Atmosphere        Beer-Lambert two-way loss, alpha = alpha_air + a*rain_rate^b
                     (same functional form and default constants as Cosys-AirSim's GPU LiDAR).
6. Receiver          Returns closer than the pulse gate (c*tau/2) merge into one echo
                     (power-weighted range). Detection: (P + N(0, sigma_n)) / sigma_n >= SNR_thr.
7. Range noise       sigma_r^2 = sigma_floor^2 + (c*t_rise/2)^2 / (2*SNR)   (timing-jitter bound),
                     so noise rises automatically as the echo gets weaker (far / dark / grazing).
8. Output            xyz (sensor frame + world frame), range, intensity, calibrated
                     reflectivity, power, SNR, ring, azimuth, firing time, ground-truth label.

Conventions
-----------
Right-handed, z up. Sensor frame: x forward, y left, z up (FLU). Orientation is Z-Y-X
(yaw, pitch, roll); positive pitch tilts the x-axis toward -z. Azimuth is measured from +x
toward +y. For AirSim-style NED use flu_to_ned() on the sensor-frame points.

Usage
-----
    python lidar_sim.py --init                 # write a documented lidar_config.jsonc next to this script
    python lidar_sim.py                        # simulate whatever lidar_config.jsonc describes
    python lidar_sim.py --config my.jsonc      # use another file (.jsonc / .json / .yaml / .toml)
    python lidar_sim.py --set sensor.rain_mm_per_h=25 --set sensor.return_mode=first
    python lidar_sim.py --selftest             # physics validation tests

Everything (sensor, pose, materials, scene objects, outputs) is read from the config file; the
Python API is unchanged:  pc = LidarSensor(cfg, pose).scan(scene);  plot_interactive(pc, scene, cfg, ...)
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

C_LIGHT = 299_792_458.0
EPS = 1e-6


# =============================================================================
# Materials
# =============================================================================
@dataclass(frozen=True)
class Material:
    """Surface reflectance at the LiDAR wavelength (values are illustrative NIR numbers)."""
    name: str
    rho_d: float                 # diffuse (Lambertian) reflectance, 0..1 (retro-reflectors may exceed 1)
    rho_s: float = 0.0           # weight of the specular lobe
    spec_exp: float = 50.0       # sharpness of the specular lobe


MATERIALS = {m.name: m for m in [
    Material("asphalt", 0.10),
    Material("concrete", 0.30),
    Material("brick", 0.25),
    Material("wood", 0.45),
    Material("aluminium", 0.55, rho_s=0.20, spec_exp=60.0),
    Material("white_paint", 0.85),
    Material("car_paint_black", 0.05, rho_s=0.30, spec_exp=120.0),
    Material("retroreflector", 4.0),
]}


# =============================================================================
# Rotation helpers
# =============================================================================
def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def flu_to_ned(xyz: np.ndarray) -> np.ndarray:
    """Forward-Left-Up -> North-East-Down style axes (x, -y, -z)."""
    return xyz * np.array([1.0, -1.0, -1.0])


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _face_ray(n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Flip normals so that they point back toward the ray origin (n . d <= 0)."""
    s = np.sign(_dot(n, d))
    s[s == 0] = 1.0
    return n * (-s)[:, None]


# =============================================================================
# Scene primitives: each returns (t, normal) for rays o + t*d with |d| = 1
# t = inf for a miss. Normals face the ray.
# =============================================================================
class Primitive:
    material: Material
    label: str

    def intersect(self, o: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def mesh(self):
        """(vertices (V,3), triangles (F,3)) used only for the interactive view; None = not drawn."""
        return None


class Plane(Primitive):
    """Infinite plane through `point` with normal `normal` (ground, wall...)."""

    def __init__(self, point, normal, material="concrete", label="plane"):
        self.p = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.n = n / np.linalg.norm(n)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        denom = d @ self.n
        num = (self.p - o) @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / denom
        ok = (np.abs(denom) > 1e-9) & (t > EPS)
        t = np.where(ok, t, np.inf)
        n = np.tile(self.n, (len(t), 1))
        return t, _face_ray(n, d)


class Sphere(Primitive):
    def __init__(self, center, radius, material="aluminium", label="sphere"):
        self.c = np.asarray(center, float)
        self.r = float(radius)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        oc = o - self.c
        b = _dot(oc, d)
        cc = _dot(oc, oc) - self.r ** 2
        disc = b * b - cc
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1, t2 = -b - sq, -b + sq
        t = np.where(t1 > EPS, t1, t2)                 # t2 only if the origin is inside
        t = np.where(ok & (t > EPS), t, np.inf)
        p = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        return t, _face_ray((p - self.c) / self.r, d)

    def mesh(self, nu=32, nv=16):
        th, ph = np.meshgrid(np.linspace(0, np.pi, nv + 1), np.linspace(0, 2 * np.pi, nu, endpoint=False), indexing="ij")
        v = self.c + self.r * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
        f = []
        for i in range(nv):
            for j in range(nu):
                a, b = i * nu + j, i * nu + (j + 1) % nu
                c, e = (i + 1) * nu + j, (i + 1) * nu + (j + 1) % nu
                f += [[a, b, e], [a, e, c]]
        return v, np.array(f)


class OrientedBox(Primitive):
    """Box with edge lengths `size` (x, y, z in the box frame), rotated by `yaw_deg` about z."""

    def __init__(self, center, size, yaw_deg=0.0, material="concrete", label="box"):
        self.c = np.asarray(center, float)
        self.h = 0.5 * np.asarray(size, float)
        self.R = rot_z(np.radians(yaw_deg))
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        lo = (o - self.c) @ self.R                      # = R^T (o - c), row-wise
        ld = d @ self.R
        ld = np.where(np.abs(ld) < 1e-12, 1e-12, ld)
        inv = 1.0 / ld
        t0, t1 = (-self.h - lo) * inv, (self.h - lo) * inv
        tmin, tmax = np.minimum(t0, t1), np.maximum(t0, t1)
        t_near, t_far = tmin.max(axis=1), tmax.min(axis=1)
        ok = (t_near <= t_far) & (t_far > EPS)
        inside = t_near <= EPS
        t = np.where(ok, np.where(inside, t_far, t_near), np.inf)
        ax = np.where(inside, tmax.argmin(axis=1), tmin.argmax(axis=1))
        idx = np.arange(len(lo))
        nl = np.zeros_like(lo)
        nl[idx, ax] = -np.sign(ld[idx, ax])
        return t, nl @ self.R.T

    def mesh(self):
        corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
        faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                 [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
        return (corners * self.h) @ self.R.T + self.c, np.array(faces)


class VerticalCylinder(Primitive):
    """Finite cylinder with its axis along +z, base centre `base`, capped (poles, trunks, tanks)."""

    def __init__(self, base, radius, height, material="wood", label="cylinder"):
        self.base = np.asarray(base, float)
        self.r = float(radius)
        self.hgt = float(height)
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.label = label

    def intersect(self, o, d):
        N = len(o)
        rel = o - self.base
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        ox, oy, oz = rel[:, 0], rel[:, 1], rel[:, 2]
        a = dx * dx + dy * dy
        b = ox * dx + oy * dy
        c = ox * ox + oy * oy - self.r ** 2
        disc = b * b - a * c
        a_ok = (a > 1e-12) & (disc >= 0)
        sq = np.sqrt(np.where(a_ok, disc, 0.0))
        a_safe = np.where(a > 1e-12, a, 1.0)
        cand = []
        for sgn in (-1.0, 1.0):                         # side wall, near/far root
            t = (-b + sgn * sq) / a_safe
            z = oz + t * dz
            cand.append(np.where(a_ok & (t > EPS) & (z >= 0) & (z <= self.hgt), t, np.inf))
        dz_ok = np.abs(dz) > 1e-12
        dz_safe = np.where(dz_ok, dz, 1.0)
        for zc in (0.0, self.hgt):                      # bottom / top cap
            t = (zc - oz) / dz_safe
            x, y = ox + t * dx, oy + t * dy
            cand.append(np.where(dz_ok & (t > EPS) & (x * x + y * y <= self.r ** 2), t, np.inf))
        T = np.stack(cand)                               # (4, N)
        k = T.argmin(axis=0)
        t = T[k, np.arange(N)]
        p = rel + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        n = np.zeros((N, 3))
        side = k < 2
        n[side, 0] = p[side, 0] / self.r
        n[side, 1] = p[side, 1] / self.r
        n[k == 2, 2] = -1.0
        n[k == 3, 2] = 1.0
        return t, _face_ray(n, d)

    def mesh(self, n=24):
        ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([self.r * np.cos(ph), self.r * np.sin(ph)], 1)
        top = self.base + [0, 0, self.hgt]
        v = np.vstack([np.c_[ring, np.zeros(n)] + self.base, np.c_[ring, np.full(n, self.hgt)] + self.base,
                       [self.base], [top]])
        f = []
        for j in range(n):
            k = (j + 1) % n
            f += [[j, k, n + k], [j, n + k, n + j], [2 * n, k, j], [2 * n + 1, n + j, n + k]]
        return v, np.array(f)


class Scene:
    def __init__(self, objects: Sequence[Primitive]):
        self.objects = list(objects)
        self.labels = [o.label for o in self.objects]
        self.rho_d = np.array([o.material.rho_d for o in self.objects] + [0.0])
        self.rho_s = np.array([o.material.rho_s for o in self.objects] + [0.0])
        self.spec_exp = np.array([o.material.spec_exp for o in self.objects] + [1.0])

    def cast(self, o: np.ndarray, d: np.ndarray):
        """Closest hit per ray -> (t, object index (-1 = miss), normal)."""
        N = len(d)
        best_t = np.full(N, np.inf)
        best_i = np.full(N, -1, dtype=np.int32)
        best_n = np.zeros((N, 3))
        for i, obj in enumerate(self.objects):
            t, n = obj.intersect(o, d)
            m = t < best_t
            best_t[m], best_i[m], best_n[m] = t[m], i, n[m]
        return best_t, best_i, best_n


# =============================================================================
# Sensor configuration (parameter names follow Cosys-AirSim where a counterpart exists)
# =============================================================================
@dataclass
class LidarConfig:
    # --- scan geometry (Cosys: NumberOfChannels, MeasurementsPerCycle, RotationsPerSecond, FOV...)
    n_channels: int = 32
    vfov_lower_deg: float = -25.0
    vfov_upper_deg: float = 15.0
    measurements_per_cycle: int = 1800           # azimuth steps per 360 deg
    hfov_start_deg: float = -180.0
    hfov_end_deg: float = 180.0
    rotations_per_s: float = 10.0
    min_range_m: float = 0.5
    max_range_m: float = 200.0                   # hard time-of-flight gate (Cosys: Range)
    # --- transmitter
    peak_power_w: float = 75.0
    beam_divergence_mrad: float = 2.0            # full angle, 1/e^2 intensity
    n_beam_samples: int = 12                     # sub-rays per pulse (1 = pencil beam)
    pulse_fwhm_ns: float = 5.0                   # echoes closer than c*tau/2 merge
    pulse_rise_ns: float = 1.0
    # --- receiver
    aperture_diam_m: float = 0.025
    optical_efficiency: float = 0.5
    noise_rms_w: float = 4.0e-9                  # receiver noise (rms, in optical-power units)
    snr_threshold: float = 10.0
    saturation_power_w: float = 5.0e-5           # power mapped to intensity = 1
    range_floor_sigma_m: float = 0.01
    noise_enabled: bool = True
    # --- atmosphere (rain constants = Cosys-AirSim defaults rainConstantA / rainConstantB)
    air_extinction_per_m: float = 2.0e-4         # ~20 km visibility
    rain_mm_per_h: float = 0.0
    rain_a: float = 0.01
    rain_b: float = 0.6
    # --- echo selection
    return_mode: str = "strongest"               # 'first' | 'strongest' | 'last'

    # derived quantities ------------------------------------------------------
    @property
    def rx_area_m2(self) -> float:
        return np.pi * (0.5 * self.aperture_diam_m) ** 2

    @property
    def extinction_per_m(self) -> float:
        rain = self.rain_a * self.rain_mm_per_h ** self.rain_b if self.rain_mm_per_h > 0 else 0.0
        return self.air_extinction_per_m + rain

    @property
    def link_const(self) -> float:               # P_t * A_r * eta
        return self.peak_power_w * self.rx_area_m2 * self.optical_efficiency

    @property
    def scan_period_s(self) -> float:
        return 1.0 / self.rotations_per_s

    def elevations_deg(self) -> np.ndarray:
        if self.n_channels == 1:                 # Cosys convention: 2D lidar uses the upper angle
            return np.array([self.vfov_upper_deg])
        return np.linspace(self.vfov_lower_deg, self.vfov_upper_deg, self.n_channels)

    def azimuths_deg(self) -> np.ndarray:
        step = 360.0 / self.measurements_per_cycle
        span = self.hfov_end_deg - self.hfov_start_deg
        n = self.measurements_per_cycle if span >= 360.0 - 1e-9 else int(np.floor(span / step + 1e-9)) + 1
        return self.hfov_start_deg + step * np.arange(n)


def received_power(cfg: LidarConfig, rho: float, cos_inc: float, R: float) -> float:
    """LiDAR equation for a Lambertian target that fills the beam."""
    return (cfg.link_const * rho / np.pi * cos_inc / R ** 2) * np.exp(-2.0 * cfg.extinction_per_m * R)


def max_range_estimate(cfg: LidarConfig, rho: float = 0.1, cos_inc: float = 1.0) -> float:
    """Range where the mean echo power equals the detection threshold (50 % detection)."""
    p_min = cfg.snr_threshold * cfg.noise_rms_w
    lo, hi = 0.05, 1.0e5
    for _ in range(200):                          # power decreases monotonically with R
        mid = np.sqrt(lo * hi)
        if received_power(cfg, rho, cos_inc, mid) > p_min:
            lo = mid
        else:
            hi = mid
    return float(lo)


@dataclass
class SensorPose:
    position: Sequence[float] = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    velocity: Sequence[float] = (0.0, 0.0, 0.0)          # world m/s, applied during the sweep
    yaw_rate_deg_s: float = 0.0


def beam_pattern(n: int, divergence_mrad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-area (sunflower) sampling of a Gaussian beam: offsets (rad) and energy weights."""
    if n <= 1:
        return np.zeros((1, 2)), np.ones(1)
    theta0 = 0.5 * divergence_mrad * 1e-3            # 1/e^2 half-angle
    sigma = theta0 / 2.0                              # I ~ exp(-2 r^2/theta0^2) = exp(-r^2/(2 sigma^2))
    r_max = 2.5 * sigma                               # contains 95.6 % of the energy
    i = np.arange(n)
    r = r_max * np.sqrt((i + 0.5) / n)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    w = np.exp(-r ** 2 / (2.0 * sigma ** 2))
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1), w / w.sum()


# =============================================================================
# Point cloud container + I/O
# =============================================================================
@dataclass
class PointCloud:
    xyz_sensor: np.ndarray          # (M,3) raw measurement r*dir in the sensor frame (shows motion skew)
    xyz_world: np.ndarray           # (M,3) positions using the true sensor pose at firing time
    range_m: np.ndarray             # measured (noisy) range
    range_true_m: np.ndarray        # noise-free echo range
    intensity: np.ndarray           # 0..1, echo power / saturation power
    reflectivity: np.ndarray        # range-compensated apparent reflectance (= rho*cos(theta)*atm loss)
    power_w: np.ndarray             # measured optical echo power
    snr: np.ndarray
    incidence_deg: np.ndarray
    ring: np.ndarray                # channel index
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    time_s: np.ndarray              # firing time within the sweep
    label_id: np.ndarray            # ground truth object index
    label_names: List[str]
    range_image: np.ndarray         # (n_channels, n_azimuth), NaN = no return
    n_beams: int

    @property
    def n(self) -> int:
        return len(self.range_m)

    def label_name(self) -> np.ndarray:
        return np.array(self.label_names)[self.label_id]

    def save_npz(self, path: str):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "label_names"}
        np.savez_compressed(path, label_names=np.array(self.label_names), **d)

    def _table(self):
        cols = [self.xyz_sensor[:, 0], self.xyz_sensor[:, 1], self.xyz_sensor[:, 2],
                self.intensity, self.reflectivity, self.range_m, self.snr,
                self.ring, self.azimuth_deg, self.time_s, self.label_id]
        names = "x y z intensity reflectivity range snr ring azimuth_deg time_s label_id".split()
        return np.column_stack(cols), names

    def save_csv(self, path: str):
        tab, names = self._table()
        np.savetxt(path, tab, delimiter=",", header=",".join(names), comments="",
                   fmt=["%.4f"] * 3 + ["%.5f", "%.5f", "%.4f", "%.2f", "%d", "%.3f", "%.6f", "%d"])

    def save_ply(self, path: str):
        """ASCII PLY (opens in CloudCompare / MeshLab / Open3D) in the sensor frame."""
        tab, _ = self._table()
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\ncomment sensor frame (x fwd, y left, z up)\n")
            f.write(f"element vertex {self.n}\n")
            for p in ("x", "y", "z", "intensity", "reflectivity"):
                f.write(f"property float {p}\n")
            f.write("property int label\nend_header\n")
            np.savetxt(f, np.column_stack([tab[:, :3], tab[:, 3], tab[:, 4], tab[:, 10]]),
                       fmt="%.4f %.4f %.4f %.5f %.5f %d")


# =============================================================================
# The sensor
# =============================================================================
class LidarSensor:
    def __init__(self, cfg: LidarConfig, pose: Optional[SensorPose] = None, seed: Optional[int] = 0):
        self.cfg = cfg
        self.pose = pose or SensorPose()
        self.rng = np.random.default_rng(seed)

    # ---- one full sweep ------------------------------------------------------
    def scan(self, scene: Scene) -> PointCloud:
        cfg, pose = self.cfg, self.pose
        el = np.radians(cfg.elevations_deg())
        az_deg = cfg.azimuths_deg()
        az = np.radians(az_deg)
        n_az, n_ch = len(az), len(el)
        AZ, EL = np.meshgrid(az, el, indexing="ij")                      # (n_az, n_ch)
        ring = np.tile(np.arange(n_ch), (n_az, 1)).ravel()
        az_deg_flat = np.degrees(AZ).ravel()
        el_deg_flat = np.degrees(EL).ravel()
        N = n_az * n_ch

        # beam direction in the sensor frame
        d_loc = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)

        # firing time of every pulse, and sensor pose at that time (rolling-scan geometry)
        t_fire = ((np.degrees(AZ) - cfg.hfov_start_deg) / 360.0 * cfg.scan_period_s).ravel()
        origin = np.asarray(pose.position, float) + np.asarray(pose.velocity, float) * t_fire[:, None]
        yaw_t = np.radians(pose.yaw_deg + pose.yaw_rate_deg_s * t_fire)
        R_mount = rot_y(np.radians(pose.pitch_deg)) @ rot_x(np.radians(pose.roll_deg))
        v = d_loc @ R_mount.T
        c, s = np.cos(yaw_t), np.sin(yaw_t)
        d_world = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)

        # ---- beam sub-rays (Gaussian profile) ----
        K = max(1, cfg.n_beam_samples)
        off, w = beam_pattern(K, cfg.beam_divergence_mrad)
        helper = np.tile([0.0, 0.0, 1.0], (N, 1))
        helper[np.abs(d_world[:, 2]) > 0.99] = [1.0, 0.0, 0.0]
        u = np.cross(d_world, helper)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        vv = np.cross(d_world, u)
        sub = d_world[:, None, :] + off[None, :, 0, None] * u[:, None, :] + off[None, :, 1, None] * vv[:, None, :]
        sub /= np.linalg.norm(sub, axis=2, keepdims=True)

        # ---- ray casting ----
        o_rep = np.repeat(origin, K, axis=0)
        d_flat = sub.reshape(-1, 3)
        t, idx, nrm = scene.cast(o_rep, d_flat)
        valid = (idx >= 0) & (t >= cfg.min_range_m) & (t <= cfg.max_range_m)

        # ---- radiometry: LiDAR equation with Lambertian + specular BRDF ----
        cos_t = np.clip(-_dot(nrm, d_flat), 0.0, 1.0)
        i_safe = np.where(idx >= 0, idx, len(scene.objects))
        cos2 = np.clip(2.0 * cos_t ** 2 - 1.0, 0.0, 1.0)                 # monostatic: lobe angle = 2*theta
        f_r = (scene.rho_d[i_safe] / np.pi
               + scene.rho_s[i_safe] * (scene.spec_exp[i_safe] + 2.0) / (2.0 * np.pi) * cos2 ** scene.spec_exp[i_safe])
        Rr = np.where(valid, t, 1.0)
        P = cfg.link_const * f_r * cos_t / Rr ** 2 * np.exp(-2.0 * cfg.extinction_per_m * Rr)
        P = np.where(valid, P, 0.0).reshape(N, K) * w[None, :]
        R = np.where(valid, t, np.inf).reshape(N, K)
        ID = idx.reshape(N, K)
        COS = cos_t.reshape(N, K)

        found, r_true, p_true, p_meas, lab, inc = self._detect(R, P, ID, COS)

        # ---- range noise from the echo SNR ----
        snr = p_meas / cfg.noise_rms_w
        r_meas = r_true.copy()
        if cfg.noise_enabled:
            sig = np.sqrt(cfg.range_floor_sigma_m ** 2
                          + (0.5 * C_LIGHT * cfg.pulse_rise_ns * 1e-9) ** 2 / (2.0 * np.maximum(snr, 1e-9)))
            r_meas = r_true + self.rng.normal(0.0, 1.0, N) * sig

        f = found
        pts_sensor = r_meas[f, None] * d_loc[f]
        pts_world = origin[f] + r_meas[f, None] * d_world[f]
        rho_app = p_meas[f] * np.pi * r_meas[f] ** 2 / cfg.link_const
        rimg = np.full(N, np.nan)
        rimg[f] = r_meas[f]
        return PointCloud(
            xyz_sensor=pts_sensor, xyz_world=pts_world, range_m=r_meas[f], range_true_m=r_true[f],
            intensity=np.clip(p_meas[f] / cfg.saturation_power_w, 0.0, 1.0), reflectivity=rho_app,
            power_w=p_meas[f], snr=snr[f], incidence_deg=np.degrees(np.arccos(np.clip(inc[f], 0, 1))),
            ring=ring[f], azimuth_deg=az_deg_flat[f], elevation_deg=el_deg_flat[f], time_s=t_fire[f],
            label_id=lab[f], label_names=list(scene.labels),
            range_image=rimg.reshape(n_az, n_ch).T, n_beams=N)

    # ---- echo formation: merge sub-ray returns inside the pulse gate, then detect ----
    def _detect(self, R, P, ID, COS):
        cfg, rng = self.cfg, self.rng
        N, K = R.shape
        order = np.argsort(R, axis=1)
        R = np.take_along_axis(R, order, 1)
        P = np.take_along_axis(P, order, 1)
        ID = np.take_along_axis(ID, order, 1)
        COS = np.take_along_axis(COS, order, 1)
        gate = 0.5 * C_LIGHT * cfg.pulse_fwhm_ns * 1e-9

        opened = np.zeros(N, bool)
        cP, cPR, cLast = np.zeros(N), np.zeros(N), np.zeros(N)
        cBestP, cBestID, cBestCos = np.zeros(N), np.full(N, -1, np.int32), np.zeros(N)
        found = np.zeros(N, bool)
        out_r, out_pt, out_pm = np.zeros(N), np.zeros(N), np.zeros(N)
        out_id, out_cos = np.full(N, -1, np.int32), np.zeros(N)

        def finalize(mask):
            sel = np.flatnonzero(mask)
            if sel.size == 0:
                return
            pt = cP[sel]
            pm = pt + (rng.normal(0.0, cfg.noise_rms_w, sel.size) if cfg.noise_enabled else 0.0)
            det = pm / cfg.noise_rms_w >= cfg.snr_threshold
            rc = cPR[sel] / np.maximum(pt, 1e-300)
            if cfg.return_mode == "first":
                take = det & ~found[sel]
            elif cfg.return_mode == "last":
                take = det
            else:
                take = det & (~found[sel] | (pm > out_pm[sel]))
            s2 = sel[take]
            out_r[s2], out_pt[s2], out_pm[s2] = rc[take], pt[take], pm[take]
            out_id[s2], out_cos[s2] = cBestID[s2], cBestCos[s2]
            found[s2] = True

        for k in range(K):
            r, p = R[:, k], P[:, k]
            hit = np.isfinite(r)
            new = hit & (~opened | ((r - cLast) > gate))
            finalize(new & opened)
            cP[new], cPR[new], cLast[new] = p[new], p[new] * r[new], r[new]
            cBestP[new], cBestID[new], cBestCos[new] = p[new], ID[new, k], COS[new, k]
            opened[new] = True
            add = hit & ~new
            cP[add] += p[add]
            cPR[add] += p[add] * r[add]
            cLast[add] = r[add]
            better = add & (p > cBestP)
            cBestP[better], cBestID[better], cBestCos[better] = p[better], ID[better, k], COS[better, k]
        finalize(opened)
        return found, out_r, out_pt, out_pm, out_id, out_cos


# =============================================================================
# Figures
# =============================================================================
def plot_cloud(pc: PointCloud, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 5.8))
    m = pc.range_m < 40
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax.scatter(*pc.xyz_sensor[m].T, c=pc.reflectivity[m], s=0.6, cmap="viridis", vmin=0, vmax=0.9)
    ax.set_box_aspect((1, 1, 0.45)); ax.view_init(24, -130)
    ax.set_xlabel("x fwd [m]"); ax.set_ylabel("y left [m]"); ax.set_zlabel("z up [m]")
    ax.set_title("3D (sensor frame, range < 40 m)\ncolour = apparent reflectivity")
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)

    ax2 = fig.add_subplot(1, 3, 2)
    cmap = plt.get_cmap("tab10")
    for lid, name in enumerate(pc.label_names):
        k = (pc.label_id == lid) & m
        if k.any():
            ax2.scatter(pc.xyz_sensor[k, 0], pc.xyz_sensor[k, 1], s=1.2, color=cmap(lid % 10), label=f"{name} ({k.sum()})")
    ax2.plot(0, 0, "k^", ms=8)
    ax2.set_aspect("equal"); ax2.set_xlabel("x fwd [m]"); ax2.set_ylabel("y left [m]")
    ax2.set_title("Top view, ground-truth labels"); ax2.legend(markerscale=6, fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(1, 3, 3)
    n_ch, n_az = pc.range_image.shape
    im = ax3.imshow(np.ma.masked_invalid(pc.range_image), aspect="auto", origin="lower", cmap="turbo", vmin=0, vmax=40,
                    extent=[-180, 180, 0, n_ch])
    ax3.set_xlabel("azimuth [deg]"); ax3.set_ylabel("channel (ring)")
    ax3.set_title("Range image (white = no return)")
    fig.colorbar(im, ax=ax3, shrink=0.8, label="range [m]")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_interactive(pc: PointCloud, scene: Scene, cfg: LidarConfig, sensor_pos: Sequence[float],
                     path: str, title: str = "") -> str:
    """Interactive Plotly view (WebGL): orbit / zoom / pan, hover read-out, colour modes, per-object filter.

    Everything (including plotly.js) is embedded, so the HTML works offline. Points are plotted in the
    WORLD frame so they line up with the scene geometry drawn as translucent meshes.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names, L = pc.label_names, len(pc.label_names)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    colour = [palette[i % len(palette)] for i in range(L)]
    label_scale = [[b, colour[i]] for i in range(L) for b in (i / L, (i + 1) / L)]
    present = [i for i in range(L) if np.any(pc.label_id == i)]

    # colour modes: name -> (per-point values, colorscale, cmin, cmax, colorbar title)
    modes = {
        "Object label (ground truth)": (pc.label_id.astype(float), label_scale, -0.5, L - 0.5, "object"),
        "Reflectivity (clipped to 0-1)": (np.clip(pc.reflectivity, 0, 1), "Viridis", 0.0, 1.0, "reflectivity"),
        "Intensity": (pc.intensity, "Viridis", 0.0, 1.0, "intensity"),
        "Range [m]": (pc.range_m, "Turbo", 0.0, float(np.percentile(pc.range_m, 99)), "range [m]"),
        "log10(SNR)": (np.log10(np.maximum(pc.snr, 1e-9)), "Plasma", float(np.log10(max(cfg.snr_threshold, 1e-9))),
                       float(np.log10(max(pc.snr.max(), 10.0))), "log10 SNR"),
        "Incidence angle [deg]": (pc.incidence_deg, "Cividis", 0.0, 90.0, "incidence [deg]"),
    }
    first = next(iter(modes))

    fig = make_subplots(rows=2, cols=1, row_heights=[0.76, 0.24], vertical_spacing=0.07,
                        specs=[[{"type": "scene"}], [{"type": "xy"}]])

    # one trace per object (so objects can be isolated) ----------------------------------
    for i in present:
        k = pc.label_id == i
        hover = np.column_stack([pc.range_m[k], pc.reflectivity[k], pc.intensity[k], pc.snr[k],
                                 pc.incidence_deg[k], pc.ring[k], pc.azimuth_deg[k]]).astype(np.float32)
        fig.add_trace(go.Scatter3d(
            x=pc.xyz_world[k, 0].astype(np.float32), y=pc.xyz_world[k, 1].astype(np.float32),
            z=pc.xyz_world[k, 2].astype(np.float32),
            mode="markers", name=names[i], customdata=hover,
            marker=dict(size=2, coloraxis="coloraxis", color=modes[first][0][k].astype(np.float32)),
            hovertemplate=("<b>%{fullData.name}</b><br>x %{x:.2f}  y %{y:.2f}  z %{z:.2f} m<br>"
                           "range %{customdata[0]:.2f} m<br>reflectivity %{customdata[1]:.3f}<br>"
                           "intensity %{customdata[2]:.3f}<br>SNR %{customdata[3]:.1f}<br>"
                           "incidence %{customdata[4]:.1f}\u00b0<br>ring %{customdata[5]:.0f}, az %{customdata[6]:.2f}\u00b0"
                           "<extra></extra>")), row=1, col=1)
    n_pts = len(present)

    sp = np.asarray(sensor_pos, float)
    fig.add_trace(go.Scatter3d(x=[sp[0]], y=[sp[1]], z=[sp[2]], mode="markers", name="sensor",
                               marker=dict(size=7, color="black", symbol="diamond"),
                               hovertemplate="LiDAR sensor<extra></extra>"), row=1, col=1)

    mesh_idx = []
    for oi, obj in enumerate(scene.objects):
        m = obj.mesh()
        if m is None:
            continue
        v, f = m
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=colour[oi % len(colour)], opacity=0.18, flatshading=True,
                                name=f"{obj.label} (true geometry)", hoverinfo="name", showscale=False), row=1, col=1)
        mesh_idx.append(len(fig.data) - 1)

    rmax = float(np.percentile(pc.range_m, 99))
    fig.add_trace(go.Heatmap(z=np.round(pc.range_image, 2), x=cfg.azimuths_deg(), y=cfg.elevations_deg(),
                             colorscale="Turbo", zmin=0, zmax=rmax, name="range image",
                             colorbar=dict(title=dict(text="range [m]"), x=1.0, y=0.115, len=0.2, thickness=14),
                             hovertemplate="az %{x:.1f}\u00b0  el %{y:.1f}\u00b0<br>range %{z:.2f} m<extra></extra>"),
                  row=2, col=1)
    fig.update_xaxes(title_text="azimuth [deg]", row=2, col=1)
    fig.update_yaxes(title_text="elevation [deg] (one row per ring)", row=2, col=1)

    def cbar(mode):
        _, scale, lo, hi, ttl = modes[mode]
        up = {"coloraxis.colorscale": scale, "coloraxis.cmin": lo, "coloraxis.cmax": hi,
              "coloraxis.colorbar.title.text": ttl,
              "coloraxis.colorbar.tickmode": "array" if mode == first else "auto"}
        if mode == first:
            up.update({"coloraxis.colorbar.tickvals": list(range(L)), "coloraxis.colorbar.ticktext": names})
        return up

    pt_idx = list(range(n_pts))
    mode_buttons = []
    for mname, (vals, *_rest) in modes.items():
        per_trace = [np.round(vals[pc.label_id == i], 2).tolist() for i in present]
        mode_buttons.append(dict(label=mname, method="update", args=[{"marker.color": per_trace}, cbar(mname), pt_idx]))

    def vis(flags):
        return dict(method="restyle", args=[{"visible": flags}, pt_idx])

    show_buttons = [dict(label="All objects", **vis([True] * n_pts))]
    for n_i, i in enumerate(present):
        show_buttons.append(dict(label=f"Only: {names[i]}", **vis([j == n_i for j in range(n_pts)])))
    if "ground" in names and names.index("ground") in present:
        g = present.index(names.index("ground"))
        show_buttons.insert(1, dict(label="Everything except ground", **vis([j != g for j in range(n_pts)])))

    geo_buttons = [dict(label="Geometry on", method="restyle", args=[{"visible": True}, mesh_idx]),
                   dict(label="Geometry off", method="restyle", args=[{"visible": False}, mesh_idx])]

    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.985, font=dict(size=14)),
        height=1000, margin=dict(l=10, r=10, t=120, b=40), paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        font=dict(color="#222"),
        scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   camera=dict(eye=dict(x=-1.5, y=-1.2, z=0.8))),
        coloraxis=dict(colorscale=modes[first][1], cmin=modes[first][2], cmax=modes[first][3],
                       colorbar=dict(title=dict(text="object"), tickmode="array", tickvals=list(range(L)), ticktext=names,
                                     x=1.0, y=0.66, len=0.55, thickness=14)),
        updatemenus=[
            dict(type="dropdown", buttons=mode_buttons, x=0.0, y=1.075, xanchor="left", yanchor="top", showactive=True),
            dict(type="dropdown", buttons=show_buttons, x=0.27, y=1.075, xanchor="left", yanchor="top", showactive=True),
            dict(type="buttons", direction="right", buttons=geo_buttons, x=0.5, y=1.075, xanchor="left", yanchor="top",
                 showactive=True, active=0),
        ],
        annotations=[dict(text=t, x=x, y=1.095, xref="paper", yref="paper", showarrow=False, xanchor="left",
                          font=dict(size=11, color="#666"))
                     for t, x in (("Colour by", 0.0), ("Show", 0.27), ("True scene geometry", 0.5))],
    )

    div = fig.to_html(full_html=False, include_plotlyjs=True,
                      config=dict(scrollZoom=True, displaylogo=False, responsive=True))
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>LiDAR point cloud (interactive)</title>
<style>
:root {{ --bg:#eef0f3; --fg:#222; box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#15171a; --fg:#ddd; }} }}
:root[data-theme="dark"] {{ --bg:#15171a; --fg:#ddd; }}
html {{ scroll-padding-top:env(safe-area-inset-top,0px); }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px system-ui,-apple-system,"Segoe UI",sans-serif; }}
.card {{ background:#fff; margin:8px; border-radius:8px; overflow:hidden; }}
.tip {{ padding:6px 14px 10px; font-size:12px; color:var(--fg); opacity:.75; }}
</style></head><body>
<div class="card">{div}</div>
<div class="tip">Drag = orbit &middot; scroll / pinch = zoom &middot; right-drag or shift-drag = pan &middot; double-click = reset view &middot;
hover a point for range, reflectivity, SNR, incidence angle &middot; drag a box on the range image to zoom it.</div>
</body></html>"""
    # plotly.js contains a literal U+FFFD; write it as a JS escape so the file is clean UTF-8 for any host
    html = html.replace("\ufffd", "\\ufffd")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


CONFIG_TEMPLATE = r'''// lidar_config.jsonc  -  every input of lidar_sim.py lives in this file.
//
//   python lidar_sim.py                                   run with this file (looked up next to the script)
//   python lidar_sim.py --config other.jsonc              run with another file
//   python lidar_sim.py --set sensor.rain_mm_per_h=25 --set sensor.return_mode=first
//
// Comments (// and /* */) are allowed. Relative paths are relative to this file's folder.
// Unknown keys are rejected with a suggestion, so a typo can never silently fall back to a default.
// Units: metres, degrees, seconds, watts unless the key name says otherwise.
// Frame: right-handed, z up. Azimuth is measured from +x toward +y.
{
  "simulation": {
    "seed": 1                                // noise seed (integer); null = different noise every run
  },

  // ---------------------------------------------------------------------------------------------
  "sensor": {
    // ---- scan geometry ----
    "n_channels": 32,                        // number of lasers (1 = 2D LiDAR fired at vfov_upper_deg)
    "vfov_lower_deg": -25.0,                 // lowest beam elevation
    "vfov_upper_deg": 15.0,                  // highest beam elevation
    "measurements_per_cycle": 1800,          // azimuth steps per 360 deg (step = 360 / N)
    "hfov_start_deg": -180.0,                // horizontal window start
    "hfov_end_deg": 180.0,                   // horizontal window end; a span >= 360 = full circle
    "rotations_per_s": 10.0,                 // head spin rate; sets pulse firing times (motion skew)
    "min_range_m": 0.5,                      // blind zone
    "max_range_m": 200.0,                    // hard time-of-flight gate
    // ---- transmitter ----
    "peak_power_w": 75.0,                    // pulse peak power
    "beam_divergence_mrad": 2.0,             // full angle at the 1/e^2 intensity points
    "n_beam_samples": 12,                    // sub-rays per pulse (1 = pencil beam, no edge effects)
    "pulse_fwhm_ns": 5.0,                    // pulse width; echoes closer than c*tau/2 merge into one
    "pulse_rise_ns": 1.0,                    // pulse rise time (enters the range-noise formula)
    // ---- receiver ----
    "aperture_diam_m": 0.025,                // receiver aperture diameter
    "optical_efficiency": 0.5,               // 0..1, lens + filter + detector
    "noise_rms_w": 4e-9,                     // receiver noise, rms optical power
    "snr_threshold": 10.0,                   // an echo is detected if (P + noise) / noise_rms >= this
    "saturation_power_w": 5e-5,              // power mapped to intensity = 1 (output scaling only)
    "range_floor_sigma_m": 0.01,             // range-noise floor reached at high SNR
    "noise_enabled": true,                   // false = exact ranges and a hard detection threshold
    // ---- atmosphere ----
    "air_extinction_per_m": 2e-4,            // roughly 3.9 / visibility  (20 km -> 2e-4, 200 m fog -> 2e-2)
    "rain_mm_per_h": 0.0,                    // rain rate; adds rain_a * rate^rain_b to the extinction
    "rain_a": 0.01,                          // Cosys-AirSim default rainConstantA
    "rain_b": 0.6,                           // Cosys-AirSim default rainConstantB
    // ---- echo selection ----
    "return_mode": "strongest"               // "first" | "strongest" | "last"
  },

  // ---------------------------------------------------------------------------------------------
  "pose": {
    "position": [0.0, 0.0, 1.8],             // sensor origin in the world frame [m]
    "roll_deg": 0.0,                         // mounting orientation, Z-Y-X order (positive pitch = nose down)
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "velocity": [0.0, 0.0, 0.0],             // world velocity during the sweep [m/s] -> rolling-scan skew
    "yaw_rate_deg_s": 0.0                    // turn rate during the sweep
  },

  // ---------------------------------------------------------------------------------------------
  "scene": {
    // Built-in materials: asphalt, concrete, brick, wood, aluminium, white_paint, car_paint_black,
    // retroreflector. Add or override your own here, e.g.
    //   "my_paint": { "rho_d": 0.4, "rho_s": 0.1, "spec_exp": 80 }
    // rho_d = diffuse reflectance, rho_s = specular lobe weight, spec_exp = lobe sharpness.
    // An object may also give an inline dict instead of a name:  "material": { "rho_d": 0.2 }
    "materials": {},

    // Object types and their keys:
    //   plane    point, normal
    //   box      center, size [x, y, z], yaw_deg (rotation about z, default 0)
    //   sphere   center, radius
    //   cylinder base (centre of the bottom cap), radius, height   (axis along +z)
    // Common optional keys: material (default "concrete"), label (default type_index), enabled (default true)
    "objects": [
      { "type": "plane",    "label": "ground",        "point": [0, 0, 0], "normal": [0, 0, 1], "material": "asphalt" },
      { "type": "box",      "label": "wall_concrete", "center": [22, -12, 3.0], "size": [12, 0.6, 6.0], "yaw_deg": 20, "material": "concrete" },
      { "type": "box",      "label": "car",           "center": [-9, 4, 0.75], "size": [4.4, 1.8, 1.5], "yaw_deg": 30, "material": "car_paint_black" },
      { "type": "box",      "label": "wall_white",    "center": [30, 12, 2.0], "size": [0.5, 14, 4.0], "yaw_deg": 0, "material": "white_paint" },
      { "type": "sphere",   "label": "sphere",        "center": [8, 3, 1.0], "radius": 1.0, "material": "aluminium" },
      { "type": "cylinder", "label": "pole",          "base": [6, -3, 0], "radius": 0.15, "height": 4.0, "material": "wood" },
      { "type": "box",      "label": "retro_box",     "center": [4, 6, 0.4], "size": [1.0, 1.0, 0.8], "yaw_deg": 15, "material": "retroreflector" }
    ]
  },

  // ---------------------------------------------------------------------------------------------
  "output": {
    "directory": "lidar_out",                // created if missing; relative to this file's folder
    "save": ["npz", "csv", "ply"],           // any subset of: npz, csv, ply
    "static_png": true,                      // matplotlib overview figure (needs matplotlib)
    "interactive_html": true,                // zoomable Plotly view (needs plotly)
    "open_browser": false,                   // open the interactive view when finished
    "title": null                            // figure title; null = auto-generated
  }
}
'''


# =============================================================================
# Config-file driven runs
# =============================================================================
class ConfigError(ValueError):
    """Raised for any problem in the configuration file (message names the offending key)."""


def _fail(where: str, msg: str):
    raise ConfigError(f"{where}: {msg}")


def _strip_jsonc(text: str) -> str:
    """Blank out // and /* */ comments (outside strings) while preserving line/column positions."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def load_config_file(path: str) -> dict:
    """Read .json / .jsonc (comments allowed), .yaml / .yml (needs PyYAML) or .toml (Python >= 3.11)."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        if ext in (".json", ".jsonc"):
            data = json.loads(_strip_jsonc(text))
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ConfigError("YAML config needs PyYAML (pip install pyyaml), or use a .jsonc file")
            data = yaml.safe_load(text)
        elif ext == ".toml":
            try:
                import tomllib
            except ImportError:
                raise ConfigError("TOML config needs Python >= 3.11, or use a .jsonc file")
            data = tomllib.loads(text)
        else:
            raise ConfigError(f"unsupported config extension '{ext}' (use .jsonc, .json, .yaml, .yml or .toml)")
    except (json.JSONDecodeError,) as e:
        raise ConfigError(f"{os.path.basename(path)}: syntax error at line {e.lineno}, column {e.colno}: {e.msg}")
    if not isinstance(data, dict):
        raise ConfigError(f"{os.path.basename(path)}: top level must be an object {{ ... }}")
    return data


def find_default_config() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("lidar_config.jsonc", "lidar_config.json", "lidar_config.yaml", "lidar_config.yml", "lidar_config.toml"):
        p = os.path.join(here, name)
        if os.path.isfile(p):
            return p
    return None


def apply_overrides(raw: dict, items: Sequence[str]) -> dict:
    """--set section.key=value  (value parsed as JSON, else taken as a plain string)."""
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set '{item}': expected KEY=VALUE, e.g. sensor.rain_mm_per_h=25")
        key, val = item.split("=", 1)
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        parts, d = key.strip().split("."), raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
            if not isinstance(d, dict):
                raise ConfigError(f"--set '{item}': '{p}' is not a section")
        d[parts[-1]] = v
    return raw


def _check_keys(d, allowed, where):
    if not isinstance(d, dict):
        _fail(where, "must be an object { ... }")
    for k in d:
        if k in allowed or str(k).startswith("_"):
            continue
        hint = difflib.get_close_matches(str(k), sorted(allowed), n=1)
        _fail(where, f"unknown key '{k}'" + (f" - did you mean '{hint[0]}'?" if hint
                                             else f" (allowed: {', '.join(sorted(allowed))})"))


def _num(v, where, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        _fail(where, f"expected a number, got {v!r}")
    if lo is not None and (v < lo or (strict_lo and v == lo)):
        _fail(where, f"must be {'>' if strict_lo else '>='} {lo}, got {v}")
    if hi is not None and v > hi:
        _fail(where, f"must be <= {hi}, got {v}")
    return v


def _vec3(v, where, nonzero=False, positive=False):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        _fail(where, f"expected a list of 3 numbers [x, y, z], got {v!r}")
    out = [float(_num(c, f"{where}[{i}]", lo=0 if positive else None, strict_lo=positive)) for i, c in enumerate(v)]
    if nonzero and not any(out):
        _fail(where, "must not be the zero vector")
    return out


_SENSOR_BOUNDS = {   # name: (lo, hi, lo_is_exclusive)
    "n_channels": (1, None, False), "measurements_per_cycle": (1, None, False), "n_beam_samples": (1, None, False),
    "rotations_per_s": (0, None, True), "min_range_m": (0, None, False), "max_range_m": (0, None, True),
    "peak_power_w": (0, None, True), "beam_divergence_mrad": (0, None, False), "pulse_fwhm_ns": (0, None, True),
    "pulse_rise_ns": (0, None, True), "aperture_diam_m": (0, None, True), "optical_efficiency": (0, 1, True),
    "noise_rms_w": (0, None, True), "snr_threshold": (0, None, True), "saturation_power_w": (0, None, True),
    "range_floor_sigma_m": (0, None, False), "air_extinction_per_m": (0, None, False),
    "rain_mm_per_h": (0, None, False), "rain_a": (0, None, False), "rain_b": (0, None, False),
}


def build_sensor(sec: Optional[dict]) -> LidarConfig:
    sec = {} if sec is None else sec
    defaults = LidarConfig()
    _check_keys(sec, {f.name for f in dataclasses.fields(LidarConfig)}, "sensor")
    vals = {}
    for k, v in sec.items():
        if str(k).startswith("_"):
            continue
        w, dflt = f"sensor.{k}", getattr(defaults, k)
        if isinstance(dflt, bool):
            if not isinstance(v, bool):
                _fail(w, f"expected true/false, got {v!r}")
        elif isinstance(dflt, str):
            if not isinstance(v, str):
                _fail(w, f"expected a string, got {v!r}")
        else:
            lo, hi, strict = _SENSOR_BOUNDS.get(k, (None, None, False))
            v = _num(v, w, lo, hi, strict)
            if isinstance(dflt, int):
                if v != int(v):
                    _fail(w, f"expected an integer, got {v}")
                v = int(v)
            else:
                v = float(v)
        vals[k] = v
    cfg = replace(defaults, **vals)
    if cfg.return_mode not in ("first", "strongest", "last"):
        _fail("sensor.return_mode", f"must be 'first', 'strongest' or 'last', got {cfg.return_mode!r}")
    if cfg.vfov_upper_deg < cfg.vfov_lower_deg:
        _fail("sensor.vfov_upper_deg", "must be >= vfov_lower_deg")
    if cfg.hfov_end_deg <= cfg.hfov_start_deg:
        _fail("sensor.hfov_end_deg", "must be > hfov_start_deg")
    if cfg.min_range_m >= cfg.max_range_m:
        _fail("sensor.min_range_m", "must be < max_range_m")
    return cfg


def build_pose(sec: Optional[dict]) -> SensorPose:
    sec = {} if sec is None else sec
    _check_keys(sec, {f.name for f in dataclasses.fields(SensorPose)}, "pose")
    d = {}
    for k, v in sec.items():
        if str(k).startswith("_"):
            continue
        d[k] = _vec3(v, f"pose.{k}") if k in ("position", "velocity") else float(_num(v, f"pose.{k}"))
    return SensorPose(**d)


def _resolve_material(m, registry, where):
    if isinstance(m, str):
        if m not in registry:
            hint = difflib.get_close_matches(m, list(registry), n=1)
            _fail(where, f"unknown material '{m}'" + (f" - did you mean '{hint[0]}'?" if hint
                                                       else f" (known: {', '.join(registry)})"))
        return registry[m]
    if isinstance(m, dict):
        _check_keys(m, {"name", "rho_d", "rho_s", "spec_exp"}, where)
        if "rho_d" not in m:
            _fail(where, "inline material needs 'rho_d'")
        return Material(str(m.get("name", "inline")), float(_num(m["rho_d"], f"{where}.rho_d", 0)),
                        float(_num(m.get("rho_s", 0.0), f"{where}.rho_s", 0)),
                        float(_num(m.get("spec_exp", 50.0), f"{where}.spec_exp", 0, strict_lo=True)))
    _fail(where, f"expected a material name or an object, got {m!r}")


_OBJECT_KEYS = {  # type: (required, optional)
    "plane": ({"point", "normal"}, set()),
    "box": ({"center", "size"}, {"yaw_deg"}),
    "sphere": ({"center", "radius"}, set()),
    "cylinder": ({"base", "radius", "height"}, set()),
}


def build_scene(sec: Optional[dict]) -> Scene:
    if not isinstance(sec, dict):
        raise ConfigError("scene: missing - define at least one object under scene.objects")
    _check_keys(sec, {"materials", "objects"}, "scene")
    registry = dict(MATERIALS)
    for name, md in (sec.get("materials") or {}).items():
        registry[name] = _resolve_material({"name": name, **md} if isinstance(md, dict) else md, registry,
                                           f"scene.materials.{name}")
    objs = sec.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ConfigError("scene.objects: must be a non-empty list")
    prims, labels = [], set()
    for i, o in enumerate(objs):
        w = f"scene.objects[{i}]"
        if not isinstance(o, dict) or "type" not in o:
            _fail(w, f"needs a 'type' ({', '.join(_OBJECT_KEYS)})")
        typ = o["type"]
        if typ not in _OBJECT_KEYS:
            hint = difflib.get_close_matches(str(typ), list(_OBJECT_KEYS), n=1)
            _fail(w, f"unknown type {typ!r}" + (f" - did you mean '{hint[0]}'?" if hint else f" (use {', '.join(_OBJECT_KEYS)})"))
        req, opt = _OBJECT_KEYS[typ]
        _check_keys(o, req | opt | {"type", "material", "label", "enabled"}, w)
        for k in req:
            if k not in o:
                _fail(w, f"{typ} needs '{k}'")
        enabled = o.get("enabled", True)
        if not isinstance(enabled, bool):
            _fail(f"{w}.enabled", "expected true/false")
        label = str(o.get("label", f"{typ}_{i}"))
        if label in labels:
            _fail(f"{w}.label", f"duplicate label '{label}'")
        labels.add(label)
        if not enabled:
            continue
        mat = _resolve_material(o.get("material", "concrete"), registry, f"{w}.material")
        if typ == "plane":
            p = Plane(_vec3(o["point"], f"{w}.point"), _vec3(o["normal"], f"{w}.normal", nonzero=True), mat, label)
        elif typ == "box":
            p = OrientedBox(_vec3(o["center"], f"{w}.center"), _vec3(o["size"], f"{w}.size", positive=True),
                            float(_num(o.get("yaw_deg", 0.0), f"{w}.yaw_deg")), mat, label)
        elif typ == "sphere":
            p = Sphere(_vec3(o["center"], f"{w}.center"), _num(o["radius"], f"{w}.radius", 0, strict_lo=True), mat, label)
        else:
            p = VerticalCylinder(_vec3(o["base"], f"{w}.base"), _num(o["radius"], f"{w}.radius", 0, strict_lo=True),
                                 _num(o["height"], f"{w}.height", 0, strict_lo=True), mat, label)
        prims.append(p)
    if not prims:
        raise ConfigError("scene.objects: every object is disabled")
    return Scene(prims)


def build_all(raw: dict):
    """Validate a raw config dict -> (LidarConfig, SensorPose, Scene, simulation dict, output dict)."""
    _check_keys(raw, {"simulation", "sensor", "pose", "scene", "output"}, "config")
    sim = raw.get("simulation") or {}
    _check_keys(sim, {"seed"}, "simulation")
    seed = sim.get("seed", 1)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        _fail("simulation.seed", "expected an integer or null")
    out = {"directory": "lidar_out", "save": ["npz", "csv", "ply"], "static_png": True,
           "interactive_html": True, "open_browser": False, "title": None}
    user_out = raw.get("output") or {}
    _check_keys(user_out, set(out), "output")
    out.update({k: v for k, v in user_out.items() if not str(k).startswith("_")})
    if not isinstance(out["directory"], str) or not out["directory"]:
        _fail("output.directory", "expected a folder name")
    if not isinstance(out["save"], list) or not set(out["save"]) <= {"npz", "csv", "ply"}:
        _fail("output.save", "expected a list containing any of: npz, csv, ply")
    for k in ("static_png", "interactive_html", "open_browser"):
        if not isinstance(out[k], bool):
            _fail(f"output.{k}", "expected true/false")
    if out["title"] is not None and not isinstance(out["title"], str):
        _fail("output.title", "expected a string or null")
    return (build_sensor(raw.get("sensor")), build_pose(raw.get("pose")), build_scene(raw.get("scene")),
            {"seed": seed}, out)


def run_from_config(raw: dict, base_dir: str, source: str = "") -> PointCloud:
    """Simulate exactly what the config describes and write every requested output."""
    cfg, pose, scene, sim, out = build_all(raw)
    out_dir = out["directory"] if os.path.isabs(out["directory"]) else os.path.join(base_dir, out["directory"])
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    pc = LidarSensor(cfg, pose, seed=sim["seed"]).scan(scene)
    dt = time.time() - t0

    print(f"config: {source or '(in memory)'}")
    print(f"scene: {len(scene.objects)} objects | scan: {pc.n_beams} pulses -> {pc.n} returns "
          f"({100 * pc.n / max(pc.n_beams, 1):.1f} %) in {dt:.2f} s")
    print(f"extinction alpha = {cfg.extinction_per_m:.2e} 1/m | return mode = {cfg.return_mode} | rain = {cfg.rain_mm_per_h} mm/h")
    for rho in (0.1, 0.3, 0.9):
        print(f"  analytic max range (normal incidence, rho={rho:.1f}): {max_range_estimate(cfg, rho):6.1f} m")
    for lid, name in enumerate(pc.label_names):
        k = pc.label_id == lid
        if k.any():
            print(f"  {name:14s} {k.sum():6d} pts   range {pc.range_m[k].min():6.1f}..{pc.range_m[k].max():6.1f} m"
                  f"   mean refl {pc.reflectivity[k].mean():.3f}")
        else:
            print(f"  {name:14s} {0:6d} pts")

    for fmt in out["save"]:
        getattr(pc, f"save_{fmt}")(os.path.join(out_dir, f"pointcloud.{fmt}"))
    used = {"_source": source, "simulation": sim, "sensor": dataclasses.asdict(cfg),
            "pose": dataclasses.asdict(pose), "scene": raw.get("scene"), "output": out}
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as fh:
        json.dump(used, fh, indent=2, default=float)

    title = out["title"] or (f"Physics-based LiDAR | {cfg.n_channels} ch | {pc.n} points | "
                             f"rain {cfg.rain_mm_per_h} mm/h | '{cfg.return_mode}' return")
    if out["static_png"]:
        try:
            plot_cloud(pc, os.path.join(out_dir, "pointcloud.png"), title)
        except ImportError:
            print("matplotlib not installed -> skipped static PNG (pip install matplotlib)")
    if out["interactive_html"]:
        try:
            html = plot_interactive(pc, scene, cfg, pose.position, os.path.join(out_dir, "pointcloud_interactive.html"), title)
            print(f"interactive view: {html}")
            if out["open_browser"]:
                import webbrowser
                webbrowser.open("file://" + os.path.abspath(html))
        except ImportError:
            print("plotly not installed -> skipped interactive view (pip install plotly)")
    print(f"outputs written to: {out_dir}")
    return pc


# =============================================================================
# Self tests: compare the simulator against closed-form physics
# =============================================================================
def selftest() -> bool:
    ok_all = True

    def report(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} {detail}")

    base = LidarConfig()

    # 1. Ground plane: range = h / sin(depression)
    h = 1.8
    cfg = replace(base, n_channels=5, vfov_lower_deg=-20, vfov_upper_deg=-6, measurements_per_cycle=1800)
    pc = LidarSensor(cfg, SensorPose(position=(0, 0, h)), seed=3).scan(Scene([Plane((0, 0, 0), (0, 0, 1), "asphalt")]))
    errs = []
    for r in range(5):
        k = pc.ring == r
        exp = h / np.sin(-np.radians(pc.elevation_deg[k]))
        errs.append(np.mean(pc.range_m[k] - exp))
    report("ground plane range vs h/sin(el)", max(abs(e) for e in errs) < 0.01,
           f"max |mean error| = {max(abs(e) for e in errs) * 1000:.2f} mm over 5 rings, {pc.n} pts")

    # 2. Sphere: points must lie on the surface
    c0, rad = np.array([10.0, 0, 1.8]), 1.0
    cfg = replace(base, n_channels=41, vfov_lower_deg=-15, vfov_upper_deg=15, hfov_start_deg=-15, hfov_end_deg=15,
                  measurements_per_cycle=3600)
    pc = LidarSensor(cfg, SensorPose(position=(0, 0, 1.8)), seed=4).scan(Scene([Sphere(c0, rad, "concrete")]))
    res = np.linalg.norm(pc.xyz_world - c0, axis=1) - rad
    report("sphere surface residual", abs(np.median(res)) < 0.005 and np.percentile(np.abs(res), 90) < 0.05,
           f"median {np.median(res) * 1000:+.2f} mm, 90th pct |res| {np.percentile(np.abs(res), 90) * 1000:.1f} mm, {pc.n} pts")

    # 3. Inverse-square law + atmosphere on a wall facing the beam (noise off -> exact)
    cfg = replace(base, n_channels=1, vfov_upper_deg=0.0, hfov_start_deg=-0.5, hfov_end_deg=0.5,
                  measurements_per_cycle=3600, noise_enabled=False)
    worst = 0.0
    for D in (5, 10, 20, 40, 80):
        pc = LidarSensor(cfg, SensorPose(position=(0, 0, 0))).scan(Scene([Plane((D, 0, 0), (-1, 0, 0), "concrete")]))
        k = np.argmin(np.abs(pc.azimuth_deg))
        th = received_power(cfg, 0.30, 1.0, pc.range_true_m[k])
        worst = max(worst, abs(pc.power_w[k] / th - 1))
    report("inverse-square + Beer-Lambert", worst < 0.005, f"max deviation from theory {worst * 100:.3f} % (5..80 m)")

    # 4. Lambert cosine law
    worst = 0.0
    for phi in (0, 30, 60, 75):
        n = (-np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0)
        pc = LidarSensor(cfg, SensorPose()).scan(Scene([Plane((10, 0, 0), n, "concrete")]))
        k = np.argmin(np.abs(pc.azimuth_deg))
        th = received_power(cfg, 0.30, np.cos(np.radians(phi)), pc.range_true_m[k])
        worst = max(worst, abs(pc.power_w[k] / th - 1))
    report("Lambert cosine law", worst < 0.01, f"max deviation from theory {worst * 100:.3f} % (0..75 deg)")

    # 5. Detection range: analytic R_max vs Monte-Carlo detection probability
    cfg = replace(base, n_channels=1, vfov_upper_deg=0.0, hfov_start_deg=-1, hfov_end_deg=1, measurements_per_cycle=36000)
    rmax = max_range_estimate(cfg, 0.10)
    probs = {}
    for f in (0.8, 1.0, 1.2):
        hits = trials = 0
        sensor = LidarSensor(cfg, SensorPose(), seed=7)
        for _ in range(5):
            pc = sensor.scan(Scene([Plane((f * rmax, 0, 0), (-1, 0, 0), replace(MATERIALS["asphalt"]))]))
            hits += pc.n
            trials += pc.n_beams
        probs[f] = hits / trials
    ok = probs[0.8] > 0.97 and 0.40 < probs[1.0] < 0.60 and probs[1.2] < 0.05
    report("detection range (rho=0.1)", ok,
           f"R_max={rmax:.1f} m; P_det at 0.8/1.0/1.2 R_max = {probs[0.8]:.3f}/{probs[1.0]:.3f}/{probs[1.2]:.3f}")

    # 6. Pulse gate / mixed pixels: two faces meeting at an edge.
    #    gap < c*tau/2 (0.75 m): sub-ray echoes merge -> intermediate "smeared" ranges appear.
    #    gap >> c*tau/2        : echoes stay separate -> every return is one face or the other.
    cfg = replace(base, n_channels=1, vfov_upper_deg=0.0, hfov_start_deg=-1, hfov_end_deg=1, measurements_per_cycle=36000,
                  noise_enabled=False)

    def step(gap):
        sc = Scene([OrientedBox((10.5, -5, 0), (1, 10, 10), 0, "concrete", "A"),
                    OrientedBox((10.5 + gap, 5, 0), (1, 10, 10), 0, "concrete", "B")])
        return LidarSensor(cfg, SensorPose()).scan(sc).range_m

    r_small, r_big = step(0.3), step(5.0)
    mid_small = np.sum((r_small > 10.01) & (r_small < 10.29))
    mid_big = np.sum((r_big > 10.1) & (r_big < 14.9))
    report("pulse-gate merging at an edge", mid_small >= 3 and mid_big == 0 and r_small.min() >= 10 - 1e-6 and r_small.max() <= (10.3 + 1e-6) / np.cos(np.radians(1.0)),
           f"gap 0.3 m -> {mid_small} merged echoes in between; gap 5 m -> {mid_big}")

    # 7. the shipped template parses and equals the code defaults
    try:
        c, p, sc, sim, out = build_all(json.loads(_strip_jsonc(CONFIG_TEMPLATE)))
        same = dataclasses.asdict(c) == dataclasses.asdict(LidarConfig()) and len(sc.objects) == 7
        report("config template == code defaults", same, f"{len(sc.objects)} objects, seed={sim['seed']}, outputs={out['save']}")
    except Exception as e:  # noqa: BLE001
        report("config template == code defaults", False, repr(e))

    # 8. bad configs are rejected with a pointer to the key
    base_raw = json.loads(_strip_jsonc(CONFIG_TEMPLATE))
    msgs = []
    for path_, val in (("sensor.n_chanels", 16), ("sensor.return_mode", "best"), ("sensor.optical_efficiency", 1.5),
                       ("scene.objects.0.material", "asphlat")):
        raw = json.loads(json.dumps(base_raw))
        d = raw
        parts = path_.split(".")
        for q in parts[:-1]:
            d = d[int(q)] if q.isdigit() else d[q]
        d[int(parts[-1]) if parts[-1].isdigit() else parts[-1]] = val
        try:
            build_all(raw)
            msgs.append(None)
        except ConfigError as e:
            msgs.append(str(e))
    report("config validation", all(m is not None for m in msgs) and "n_channels" in msgs[0] and "asphalt" in msgs[3],
           f"{sum(m is not None for m in msgs)}/4 bad configs rejected; e.g. {msgs[0]}")

    print("ALL TESTS PASSED" if ok_all else "SOME TESTS FAILED")
    return ok_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Physics-based LiDAR simulator driven by a config file")
    ap.add_argument("--config", help="config file (.jsonc/.json/.yaml/.toml); default: lidar_config.* next to this script")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one value, e.g. --set sensor.rain_mm_per_h=25 (repeatable)")
    ap.add_argument("--init", action="store_true", help="write a fully documented template config and exit")
    ap.add_argument("--force", action="store_true", help="allow --init to overwrite an existing file")
    ap.add_argument("--selftest", action="store_true", help="run the physics validation tests")
    a = ap.parse_args(argv)
    try:
        if a.selftest:
            return 0 if selftest() else 1
        if a.init:
            target = a.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_config.jsonc")
            if os.path.exists(target) and not a.force:
                print(f"{target} already exists (use --force to overwrite)")
                return 1
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(CONFIG_TEMPLATE)
            print(f"wrote template config: {target}")
            return 0
        path = a.config or find_default_config()
        if path is None:
            raise ConfigError("no config file found next to the script; create one with:  python lidar_sim.py --init")
        if not os.path.isfile(path):
            raise ConfigError(f"config file not found: {path}")
        raw = apply_overrides(load_config_file(path), a.set)
        run_from_config(raw, os.path.dirname(os.path.abspath(path)), source=path)
        return 0
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
