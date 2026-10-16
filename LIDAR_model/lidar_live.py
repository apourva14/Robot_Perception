#!/usr/bin/env python3
"""
lidar_live.py - live window for the LiDAR-in-the-loop navigation of lidar_nav.py
================================================================================

    python lidar_live.py                       clear weather
    python lidar_live.py --scenario dense_fog  start in fog (names from nav_config.jsonc)
    python lidar_live.py --speed 2             run twice as fast as real time (0 = as fast as possible)

The simulation is exactly lidar_nav.run(); this script only draws it while it runs.

  left    top view: occupancy map, perceived obstacle points (orange), ground returns (blue),
          free-path fan (green), detection range R_max (dotted), camera field of view (yellow)
  right   first-person view from the robot: ray-cast render of the scene with the same
          extinction (fog / rain) as the LiDAR, LiDAR returns overlaid and colored by range
  bottom  true vs perceived distance to the nearest obstacle, and speed

Keys:  space pause/resume  |  1 2 3 ... restart in scenario n  |  r restart  |  f follow robot / whole scene
       l LiDAR overlay on/off  |  m occupancy map on/off  |  q quit

Needs numpy + matplotlib with a GUI backend (the default on Windows/macOS).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lidar_nav as ln  # noqa: E402
import lidar_sim as ls  # noqa: E402

# visible-light albedo per material, for the first-person render only
ALBEDO = {"asphalt": (0.30, 0.30, 0.32), "concrete": (0.66, 0.64, 0.60), "brick": (0.62, 0.32, 0.25),
          "wood": (0.60, 0.42, 0.24), "aluminium": (0.80, 0.82, 0.88), "white_paint": (0.95, 0.95, 0.95),
          "car_paint_black": (0.07, 0.07, 0.09), "retroreflector": (0.95, 0.78, 0.10)}
SKY = np.array([0.62, 0.78, 0.95])
FOG = np.array([0.80, 0.82, 0.85])


class FirstPerson:
    """Pinhole camera at the sensor, looking along the robot heading; rendered by ray casting."""

    def __init__(self, scene, eye_h, w=240, h=135, hfov_deg=90.0, pitch_deg=-6.0):
        self.scene, self.eye_h, self.w, self.h = scene, eye_h, w, h
        self.f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2)
        self.hfov = np.radians(hfov_deg)
        u = (w / 2 - (np.arange(w) + 0.5)) / self.f              # + = left
        v = (h / 2 - (np.arange(h) + 0.5)) / self.f              # + = up
        U, V = np.meshgrid(u, v)
        p = np.radians(pitch_deg)
        # camera frame (fwd, left, up) tilted by pitch
        self.d_cam = np.stack([np.cos(p) - V * np.sin(p), U, np.sin(p) + V * np.cos(p)], -1).reshape(-1, 3)
        self.d_cam /= np.linalg.norm(self.d_cam, axis=1, keepdims=True)
        self.pitch = p
        self.albedo = np.array([ALBEDO.get(o.material.name, (0.6, 0.6, 0.6)) for o in scene.objects] + [(0, 0, 0)])
        self.is_ground = np.array([isinstance(o, ls.Plane) for o in scene.objects] + [False])
        self.sun = np.array([0.4, 0.3, 0.87]) / np.linalg.norm([0.4, 0.3, 0.87])

    def _rot(self, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def render(self, x, y, th, alpha):
        R = self._rot(th)
        d = self.d_cam @ R.T
        o = np.tile([x, y, self.eye_h], (len(d), 1))
        t, idx, n = self.scene.cast(o, d)
        hit = idx >= 0
        i = np.where(hit, idx, len(self.scene.objects))
        col = self.albedo[i].copy()
        p = o + np.where(hit, t, 0)[:, None] * d
        g = hit & self.is_ground[i]                               # 1 m tiles on the ground for a motion cue
        tile = ((np.floor(p[:, 0]) + np.floor(p[:, 1])) % 2 == 0)
        col[g] *= np.where(tile[g], 1.0, 0.82)[:, None]
        shade = 0.35 + 0.65 * np.clip(np.einsum("ij,j->i", n, self.sun), 0, 1)
        col = col * np.where(hit, shade, 1.0)[:, None]
        col[~hit] = SKY - 0.25 * np.clip(d[~hit, 2], 0, 1)[:, None]
        # same Beer-Lambert extinction as the LiDAR (one way, eye -> surface)
        tr = np.exp(-alpha * np.where(hit, t, 300.0))
        col = col * tr[:, None] + FOG * (1 - tr)[:, None]
        depth = np.where(hit, t, np.inf)
        return np.clip(col, 0, 1).reshape(self.h, self.w, 3), depth.reshape(self.h, self.w)

    def project(self, pts, x, y, th, depth):
        """World points -> pixel coordinates (only points in view and not hidden behind the render)."""
        R = self._rot(th)
        rel = (pts - [x, y, self.eye_h]) @ R                      # robot frame (fwd, left, up)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        fwd = rel[:, 0] * cp + rel[:, 2] * sp
        up = -rel[:, 0] * sp + rel[:, 2] * cp
        ok = fwd > 0.3
        col = self.w / 2 - self.f * rel[:, 1] / np.where(ok, fwd, 1)
        row = self.h / 2 - self.f * up / np.where(ok, fwd, 1)
        ok &= (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        ci, ri = col[ok].astype(int), row[ok].astype(int)
        dist = np.linalg.norm(rel[ok], axis=1)
        vis = dist <= depth[ri, ci] * 1.02 + 0.15
        sel = np.flatnonzero(ok)[vis]
        return col[sel], row[sel], np.linalg.norm(rel[sel], axis=1)


class LiveView:
    def __init__(self, nav, speed):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        self.plt, self.LC = plt, LineCollection
        self.nav, self.speed = nav, speed
        self.names = [s["name"] for s in nav["scenarios"]]
        self.paused = self.follow = False
        self.show_lidar = self.show_map = True
        self.request = None                                       # scenario index to switch to, or "quit"
        self.fig = None

    # ---- figure (rebuilt for every scenario) ----
    def setup(self, res0):
        plt = self.plt
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(15, 8.2))
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "request", "quit"))
        fig = self.fig
        fig.clf()
        gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], width_ratios=[1.15, 1], hspace=0.28, wspace=0.12,
                              left=0.05, right=0.98, top=0.93, bottom=0.07)
        self.ax_top, self.ax_cam = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        self.ax_d = fig.add_subplot(gs[1, :])
        nav, scene, cfg, grid = self.nav, res0["scene"], res0["cfg"], res0["grid"]
        rb = nav["robot"]
        self.alpha = cfg.extinction_per_m
        self.rmax = ls.max_range_estimate(cfg, 0.1)
        self.cam = FirstPerson(scene, nav["mount_height_m"])

        a = self.ax_top
        self.im_map = a.imshow(grid.prob(), origin="lower", extent=[grid.x0, grid.x1, grid.y0, grid.y1],
                               cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.5, zorder=0)
        ln.draw_scene(a, scene, nav)
        self.sc_gnd, = a.plot([], [], ".", color="#6baed6", ms=1, zorder=1)
        self.sc_obs, = a.plot([], [], ".", color="#e6550d", ms=2.5, zorder=6)
        self.fan = self.LC([], colors="#31a354", linewidths=0.6, alpha=0.6, zorder=2)
        a.add_collection(self.fan)
        self.fov_c = plt.Circle((0, 0), 1, fill=False, ls=":", color="purple", lw=1.2, zorder=2)
        a.add_patch(self.fov_c)
        self.cam_fov, = a.plot([], [], "-", color="gold", lw=1.2, zorder=7)
        self.trail, = a.plot([], [], "k-", lw=1.2, zorder=5)
        self.body = plt.Circle((0, 0), rb["radius_m"], color="k", zorder=8)
        a.add_patch(self.body)
        self.head, = a.plot([], [], "-", color="yellow", lw=2.5, zorder=9)
        self.goal_dir, = a.plot([], [], "-", color="#00c000", lw=2, zorder=9)
        self.hud = a.text(0.01, 0.99, "", transform=a.transAxes, va="top", fontsize=8, family="monospace",
                          bbox=dict(fc="w", alpha=0.8, lw=0), zorder=10)
        self.set_view(rb["start"])

        c = self.ax_cam
        img, _ = self.cam.render(rb["start"][0], rb["start"][1], np.radians(rb["start_heading_deg"]), self.alpha)
        self.im_cam = c.imshow(img, extent=[0, self.cam.w, self.cam.h, 0], interpolation="bilinear")
        self.sc_cam = c.scatter([], [], s=4, c=[], cmap="turbo", vmin=0, vmax=min(self.rmax, 30), zorder=3)
        c.set_xlim(0, self.cam.w)
        c.set_ylim(self.cam.h, 0)
        c.set_xticks([])
        c.set_yticks([])
        self.cam_txt = c.text(0.01, 0.02, "", transform=c.transAxes, color="w", fontsize=8,
                              bbox=dict(fc="k", alpha=0.5, lw=0))
        cb = fig.colorbar(self.sc_cam, ax=c, orientation="horizontal", fraction=0.05, pad=0.02)
        cb.set_label("LiDAR return range [m]", fontsize=8)
        c.set_title("first-person view (render + live LiDAR returns)", fontsize=10)

        d = self.ax_d
        self.l_true, = d.plot([], [], "k-", lw=1.5, label="true distance d")
        self.l_perc, = d.plot([], [], ".", color="#e6550d", ms=3, label="perceived d-hat (LiDAR)")
        self.l_v, = d.plot([], [], "-", color="#3182bd", lw=1, label="speed v [m/s]")
        d.axhline(rb["radius_m"], color="r", ls="--", lw=1, label="collision")
        d.set_xlim(0, rb["max_steps"] * rb["dt"] * 0.35)
        d.set_ylim(0, 6)
        d.set_xlabel("time [s]")
        d.legend(fontsize=7, loc="upper right", ncol=4)
        fig.suptitle(f"scenario '{self.names[self.current]}'   alpha = {self.alpha:.3g} 1/m   R_max(rho=0.1) = {self.rmax:.1f} m"
                     f"        keys: space pause | {' '.join(str(i + 1) for i in range(len(self.names)))} scenario"
                     f" ({', '.join(self.names)}) | r restart | f follow | l lidar | m map | q quit", fontsize=9)

    def set_view(self, xy):
        if self.follow:
            self.ax_top.set_xlim(xy[0] - 9, xy[0] + 9)
            self.ax_top.set_ylim(xy[1] - 6.5, xy[1] + 6.5)
        else:
            self.ax_top.set_xlim(-5, 32)
            self.ax_top.set_ylim(-9, 9)

    def on_key(self, e):
        k = e.key
        if k == " ":
            self.paused = not self.paused
        elif k == "q":
            self.request = "quit"
        elif k == "r":
            self.request = self.current
        elif k and k.isdigit() and 1 <= int(k) <= len(self.names):
            self.request = int(k) - 1
        elif k == "f":
            self.follow = not self.follow
        elif k == "l":
            self.show_lidar = not self.show_lidar
        elif k == "m":
            self.show_map = not self.show_map

    # ---- called by lidar_nav.run() after every control step ----
    def on_step(self, st):
        t0 = time.time()
        if st["k"] == 0:
            self.setup(st)
        lg = np.array(st["log"])
        t, x, y, th, v, w, dtrue, dperc = lg[-1, :8]
        rb = self.nav["robot"]

        self.im_map.set_data(st["grid"].prob())
        self.im_map.set_visible(self.show_map)
        self.sc_obs.set_data(st["obs"][:, 0], st["obs"][:, 1])
        self.sc_gnd.set_data(st["gnd"][::3, 0], st["gnd"][::3, 1])
        if st["L"] is not None:
            H, L = st["H"], st["L"]
            self.fan.set_segments([[(x, y), (x + l * np.cos(h), y + l * np.sin(h))] for h, l in zip(H, L)])
        self.fov_c.center, self.fov_c.radius = (x, y), min(self.rmax, 60)
        hf = self.cam.hfov / 2
        r = 6.0
        self.cam_fov.set_data([x + r * np.cos(th + hf), x, x + r * np.cos(th - hf)],
                              [y + r * np.sin(th + hf), y, y + r * np.sin(th - hf)])
        self.trail.set_data(lg[:, 1], lg[:, 2])
        self.body.center = (x, y)
        self.head.set_data([x, x + 0.9 * np.cos(th)], [y, y + 0.9 * np.sin(th)])
        self.goal_dir.set_data([x, x + 1.5 * np.cos(st["psi"])], [y, y + 1.5 * np.sin(st["psi"])])
        self.set_view((x, y))
        self.hud.set_text(f"t = {t:5.1f} s   v = {v:4.2f} m/s   omega = {np.degrees(w):6.1f} deg/s\n"
                          f"true clearance      {dtrue - rb['radius_m']:5.2f} m (surface gap)\n"
                          f"perceived clearance {dperc - rb['radius_m']:5.2f} m\n"
                          f"returns {st['pc'].n:5d}  obstacle pts {len(st['obs']):5d}   "
                          f"{'PAUSED' if self.paused else st['status']}")

        img, depth = self.cam.render(x, y, th, self.alpha)
        self.im_cam.set_data(img)
        if self.show_lidar:
            u, vv, rr = self.cam.project(st["pc"].xyz_world, x, y, th, depth)
            self.sc_cam.set_offsets(np.column_stack([u, vv]))
            self.sc_cam.set_array(rr)
            self.sc_cam.set_sizes(np.clip(40.0 / np.maximum(rr, 1), 1, 12))
        else:
            self.sc_cam.set_offsets(np.zeros((0, 2)))
        self.cam_txt.set_text(f"nearest perceived obstacle: {dperc:.2f} m   "
                              f"visibility ~ {min(3.9 / self.alpha, 9999):.0f} m")

        self.l_true.set_data(lg[:, 0], lg[:, 6])
        self.l_perc.set_data(lg[:, 0], np.where(np.isfinite(lg[:, 7]), lg[:, 7], np.nan))
        self.l_v.set_data(lg[:, 0], lg[:, 4])
        if t > self.ax_d.get_xlim()[1] - 1:
            self.ax_d.set_xlim(0, t + 10)

        # pacing: real time * speed, keep the window responsive while paused
        wait = (rb["dt"] / self.speed if self.speed > 0 else 0.0) - (time.time() - t0)
        self.plt.pause(max(wait, 0.001))
        while self.paused and self.request is None:
            self.plt.pause(0.05)
        return self.request is None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "nav_config.jsonc"))
    ap.add_argument("--scenario", default=None, help="scenario to start with (default: first)")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 2 = twice as fast, 0 = no waiting")
    ap.add_argument("--snapshot", type=int, default=None, help=argparse.SUPPRESS)  # save frame N to PNG and exit
    a = ap.parse_args(argv)
    import matplotlib.pyplot as plt
    nav = ls.load_config_file(a.config)
    view = LiveView(nav, a.speed)
    idx = view.names.index(a.scenario) if a.scenario else 0

    if a.snapshot is not None:                                    # headless check / figure for the report
        orig = view.on_step

        def snap(st):
            orig(st)
            if st["k"] == a.snapshot:
                view.fig.savefig(os.path.join(HERE, nav["output"]["directory"], f"live_{view.names[idx]}.png"), dpi=110)
                return False
            return True
        view.on_step = snap

    plt.ion()
    while True:
        view.current, view.request = idx, None
        res = ln.run(nav, nav["scenarios"][idx], on_step=view.on_step)
        if a.snapshot is not None or view.request == "quit":
            break
        if view.request is None:                                  # finished: wait for a key or window close
            view.hud.set_text(view.hud.get_text() + f"\n{res['status'].upper()} - press r / 1-{len(view.names)} or q")
            while view.request is None:
                plt.pause(0.1)
            if view.request == "quit":
                break
        idx = view.request
    plt.close("all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
