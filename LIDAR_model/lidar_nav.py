#!/usr/bin/env python3
"""
lidar_nav.py - LiDAR-in-the-loop navigation built on lidar_sim.py
=================================================================

A unicycle robot drives from a start to a goal through the lidar_sim.py scene. Every control step:

1. Sense      a full physics-based LiDAR sweep (lidar_sim.LidarSensor) from the robot's current pose,
              including its velocity and turn rate during the sweep (rolling-scan skew).
2. Perceive   ground removal by height only (z < z_ground) -> obstacle points. Ground-truth labels are
              NEVER used for control; they are only used afterwards to score the perception.
3. Plan       free-path heading histogram (VFH-like): for every candidate heading, the distance a disc of
              radius r + margin can travel before touching a perceived point. Cost = goal alignment +
              turning effort - free space; speed is limited so the robot never plans closer than d_stop.
4. Map        Bayesian log-odds occupancy grid: obstacle returns = occupied, the 2D line of sight to every
              return and every ground return = free.
5. Score      true clearance (analytic distance to the obstacle footprints) vs perceived clearance
              (nearest perceived point), and the range at which each obstacle is first detected.

The same run is repeated for each environmental scenario (clear / rain / fog) in nav_config.jsonc.

Usage:   python lidar_nav.py [--config nav_config.jsonc] [--scenario NAME] [--no-gif]
Needs:   numpy, matplotlib  (pillow for the GIF animation)    ->  pip install numpy matplotlib pillow
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
import warnings

import numpy as np

# numpy 2.0.x on macOS (Accelerate BLAS) emits spurious "encountered in matmul" warnings; results are unaffected.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lidar_sim as ls  # noqa: E402


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
# Ground truth geometry (used only for scoring, never by the robot)
# =============================================================================
def footprint_distance(obj, xy, z_hi):
    """Horizontal distance from point xy to the obstacle footprint; inf if the obstacle is entirely
    above the robot (z > z_hi) or is the ground plane."""
    xy = np.asarray(xy, float)
    if isinstance(obj, ls.OrientedBox):
        if obj.c[2] - obj.h[2] > z_hi:
            return np.inf
        q = obj.R[:2, :2].T @ (xy - obj.c[:2])
        return float(np.linalg.norm(np.maximum(np.abs(q) - obj.h[:2], 0.0)))
    if isinstance(obj, ls.VerticalCylinder):
        if obj.base[2] > z_hi:
            return np.inf
        return max(float(np.linalg.norm(xy - obj.base[:2])) - obj.r, 0.0)
    if isinstance(obj, ls.Sphere):
        if obj.c[2] - obj.r > z_hi:
            return np.inf
        return max(float(np.linalg.norm(xy - obj.c[:2])) - obj.r, 0.0)
    return np.inf                                    # plane (ground)


def footprint_outline(obj):
    """Closed 2D polygon of an obstacle footprint, for drawing."""
    if isinstance(obj, ls.OrientedBox):
        h = obj.h[:2]
        loc = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]], [-h[0], -h[1]]])
        return loc @ obj.R[:2, :2].T + obj.c[:2]
    if isinstance(obj, (ls.VerticalCylinder, ls.Sphere)):
        c = obj.base[:2] if isinstance(obj, ls.VerticalCylinder) else obj.c[:2]
        a = np.linspace(0, 2 * np.pi, 40)
        return c + obj.r * np.stack([np.cos(a), np.sin(a)], 1)
    return None


# =============================================================================
# Perception and planning
# =============================================================================
def split_returns(pc, z_ground, z_top):
    """Height-based segmentation of a sweep: (obstacle xy, ground xy, obstacle labels for scoring)."""
    z = pc.xyz_world[:, 2]
    obs = (z >= z_ground) & (z <= z_top)
    gnd = z < z_ground
    return pc.xyz_world[obs, :2], pc.xyz_world[gnd, :2], pc.label_id[obs]


def free_path_lengths(pts, xy, headings, r_inf, lookahead):
    """Distance a disc of radius r_inf can translate from xy along each heading before touching a point."""
    L = np.full(len(headings), float(lookahead))
    if len(pts) == 0:
        return L
    q = pts - xy                                                     # (M,2)
    u = np.stack([np.cos(headings), np.sin(headings)], 1)            # (H,2)
    along = q @ u.T                                                  # (M,H)
    perp = np.abs(q[:, 0, None] * u[None, :, 1] - q[:, 1, None] * u[None, :, 0])
    hit = (perp < r_inf) & (along > -r_inf)
    contact = along - np.sqrt(np.maximum(r_inf ** 2 - perp ** 2, 0.0))
    contact = np.where(hit, contact, np.inf)
    return np.clip(contact.min(axis=0), 0.0, lookahead)


class OccupancyGrid:
    """Log-odds occupancy grid, l <- l + l_inv(z) (binary Bayes filter with a static-world prior 0.5)."""

    def __init__(self, g):
        self.res = g["resolution_m"]
        self.x0, self.x1, self.y0, self.y1 = g["extent"]
        self.nx = int(round((self.x1 - self.x0) / self.res))
        self.ny = int(round((self.y1 - self.y0) / self.res))
        self.l = np.zeros((self.ny, self.nx))
        self.l_occ, self.l_free, self.l_clamp = g["l_occ"], g["l_free"], g["l_clamp"]

    def _cells(self, xy):
        ix = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.unique(iy[ok] * self.nx + ix[ok])

    def update(self, origin, obs, gnd, rng, max_rays=3000):
        ends = np.vstack([obs, gnd])
        if len(ends) > max_rays:
            ends = ends[rng.choice(len(ends), max_rays, replace=False)]
        free = [gnd]
        if len(ends):
            d = np.linalg.norm(ends - origin, axis=1)
            n = int(np.ceil(d.max() / (0.5 * self.res)))
            s = np.linspace(0.0, 1.0, n, endpoint=False)[None, :, None]
            ray = origin + s * (ends - origin)[:, None, :]               # samples up to (not incl.) the end
            keep = (s[..., 0] * d[:, None]) < (d[:, None] - self.res)
            free.append(ray[keep])
        occ = self._cells(obs) if len(obs) else np.array([], int)
        fr = np.setdiff1d(self._cells(np.vstack(free)), occ)
        flat = self.l.ravel()
        flat[fr] += self.l_free
        flat[occ] += self.l_occ
        np.clip(self.l, -self.l_clamp, self.l_clamp, out=self.l)

    def prob(self):
        return 1.0 / (1.0 + np.exp(-self.l))


# =============================================================================
# One navigation run
# =============================================================================
def build_world(nav, scen):
    base = ls.load_config_file(os.path.join(HERE, nav["base_config"]))
    raw = copy.deepcopy(base)
    raw["sensor"] = {**raw.get("sensor", {}), **nav.get("sensor_overrides", {}), **scen.get("sensor", {})}
    raw["scene"]["objects"] = list(raw["scene"]["objects"]) + list(nav.get("extra_objects", []))
    cfg, _, scene, _, _ = ls.build_all(raw)
    return cfg, scene


def run(nav, scen, verbose=True, on_step=None):
    """Simulate one scenario. on_step(state) is called after every control step (live viewer);
    returning False from it stops the run."""
    cfg, scene = build_world(nav, scen)
    rb, pl = nav["robot"], nav["planner"]
    seed = nav["output"].get("seed", 1)
    sensor = ls.LidarSensor(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid(nav["grid"])

    x, y = map(float, rb["start"])
    th = np.radians(rb["start_heading_deg"])
    goal = np.asarray(rb["goal"], float)
    r_inf = rb["radius_m"] + pl["safety_margin_m"]
    dt, wmax = rb["dt"], np.radians(rb["omega_max_deg_s"])
    H = np.linspace(-np.pi, np.pi, pl["n_headings"], endpoint=False)
    v = w = 0.0
    obstacle_ids = [i for i, o in enumerate(scene.objects) if not isinstance(o, ls.Plane)]
    first_det = {scene.labels[i]: np.nan for i in obstacle_ids}
    log, frames = [], []
    status = "timeout"

    for k in range(rb["max_steps"]):
        # ---- 1. sense (pose and motion during the sweep come from the robot state) ----
        sensor.pose = ls.SensorPose(position=(x, y, nav["mount_height_m"]), yaw_deg=np.degrees(th),
                                    velocity=(v * np.cos(th), v * np.sin(th), 0.0), yaw_rate_deg_s=np.degrees(w))
        pc = sensor.scan(scene)
        # ---- 2. perceive ----
        obs, gnd, obs_lab = split_returns(pc, pl["z_ground_m"], rb["height_m"])
        grid.update(np.array([x, y]), obs, gnd, rng)
        # ---- scoring only: true clearance and first-detection ranges ----
        d_true = {scene.labels[i]: footprint_distance(scene.objects[i], (x, y), rb["height_m"]) for i in obstacle_ids}
        for i in obstacle_ids:
            lab = scene.labels[i]
            if np.isnan(first_det[lab]) and np.count_nonzero(obs_lab == i) >= 3:
                first_det[lab] = d_true[lab]
        true_clear = min(d_true.values())
        perc_clear = float(np.min(np.linalg.norm(obs - [x, y], axis=1))) if len(obs) else np.inf
        # ---- 3. plan ----
        to_goal = goal - [x, y]
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal < rb["goal_tolerance_m"]:
            status = "goal"
            log.append((k * dt, x, y, th, 0.0, 0.0, true_clear, perc_clear, pc.n, len(obs), dist_goal))
            if on_step:
                on_step(dict(k=k, log=log, pc=pc, obs=obs, gnd=gnd, L=None, psi=th, grid=grid, cfg=cfg,
                             scene=scene, H=H, status=status))
            break
        psi_goal = np.arctan2(to_goal[1], to_goal[0])
        L = free_path_lengths(obs, np.array([x, y]), H, r_inf, pl["lookahead_m"])
        need = min(pl["d_stop_m"] + 0.5, dist_goal)
        J = (pl["w_goal"] * np.abs(wrap(H - psi_goal)) + pl["w_turn"] * np.abs(wrap(H - th))
             - pl["w_free"] * L / pl["lookahead_m"])
        J = np.where(L >= need, J, np.inf)
        best = int(np.argmin(J)) if np.isfinite(J).any() else int(np.argmax(L))
        psi = H[best]
        w = float(np.clip(pl["k_omega"] * wrap(psi - th), -wmax, wmax))
        L_now = free_path_lengths(obs, np.array([x, y]), np.array([th]), r_inf, pl["lookahead_m"])[0]
        scale = np.clip((L_now - pl["d_stop_m"]) / (pl["d_slow_m"] - pl["d_stop_m"]), 0.0, 1.0)
        v = float(rb["v_max"] * scale * max(np.cos(wrap(psi - th)), 0.0))
        v = min(v, max(L_now - pl["d_stop_m"], 0.0) / dt, dist_goal / dt)   # safety bound (Theorem 3)
        log.append((k * dt, x, y, th, v, w, true_clear, perc_clear, pc.n, len(obs), dist_goal))
        frames.append(dict(obs=obs, gnd=gnd[::4], L=L, psi=psi))
        if on_step and on_step(dict(k=k, log=log, pc=pc, obs=obs, gnd=gnd, L=L, psi=psi, grid=grid, cfg=cfg,
                                    scene=scene, H=H, status="running")) is False:
            status = "stopped"
            break
        if true_clear < rb["radius_m"]:
            status = "collision"
            break
        # ---- 4. move: unicycle, forward Euler (translate, then rotate) ----
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th = float(wrap(th + w * dt))

    log = np.array(log)
    res = dict(name=scen["name"], cfg=cfg, scene=scene, log=log, frames=frames, grid=grid,
               first_det=first_det, status=status, H=H)
    if verbose:
        steps = len(log)
        path = float(np.sum(np.linalg.norm(np.diff(log[:, 1:3], axis=0), axis=1)))
        print(f"[{scen['name']:>10}] {status:9s} steps={steps:3d}  t={log[-1, 0]:5.1f} s  path={path:5.1f} m  "
              f"min true clearance={log[:, 6].min() - rb['radius_m']:.2f} m (surface gap)  "
              f"alpha={cfg.extinction_per_m:.3f} 1/m  returns/scan={log[:, 8].mean():.0f}")
    return res


# =============================================================================
# Figures
# =============================================================================
COLORS = {"clear": "#1f77b4", "heavy_rain": "#2ca02c", "dense_fog": "#d62728"}


def _color(name, i):
    return COLORS.get(name, f"C{i}")


def draw_scene(ax, scene, nav):
    for o in scene.objects:
        poly = footprint_outline(o)
        if poly is not None:
            ax.fill(poly[:, 0], poly[:, 1], color="0.35", alpha=0.85, zorder=3)
            c = poly[:-1].mean(0)
            ax.text(c[0], c[1], o.label, fontsize=6, ha="center", va="center", color="w", zorder=4, clip_on=True)
    s, g = nav["robot"]["start"], nav["robot"]["goal"]
    ax.plot(*s, "ko", ms=6, zorder=5)
    ax.plot(*g, "k*", ms=12, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_trajectories(results, nav, path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(results), 1, figsize=(10, 3.6 * len(results)), squeeze=False, layout="constrained")
    for ax, ref in zip(axs[:, 0], results):
        g = ref["grid"]
        im = ax.imshow(g.prob(), origin="lower", extent=[g.x0, g.x1, g.y0, g.y1], cmap="RdYlGn_r", vmin=0, vmax=1,
                       alpha=0.55)
        draw_scene(ax, ref["scene"], nav)
        for i, r in enumerate(results):
            own = r is ref
            ax.plot(r["log"][:, 1], r["log"][:, 2], "-", lw=2.2 if own else 0.8, alpha=1 if own else 0.6,
                    color=_color(r["name"], i), label=f"{r['name']} ({r['status']}, {r['log'][-1, 0]:.0f} s)")
        ax.set_xlim(-5, 32)
        ax.set_ylim(-7, 8)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(f"Occupancy map built in '{ref['name']}' (green = free, red = occupied, yellow = unknown)",
                     fontsize=9)
    fig.colorbar(im, ax=axs[:, 0].tolist(), shrink=0.6, label="P(occupied)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clearance(results, nav, path):
    import matplotlib.pyplot as plt
    rr = nav["robot"]["radius_m"]
    n = len(results)
    fig, axs = plt.subplots(n + 1, 1, figsize=(9, 2.3 * (n + 1)), sharex=True, squeeze=False)
    for i, r in enumerate(results):
        lg = r["log"]
        axs[n, 0].plot(lg[:, 0], 100 * (lg[:, 7] - lg[:, 6]), ".-", lw=0.6, ms=3, color=_color(r["name"], i),
                       label=r["name"])
    axs[n, 0].axhline(0, color="k", lw=0.8)
    axs[n, 0].set_ylim(-8, 8)
    axs[n, 0].set_ylabel(r"$\hat d - d$ [cm]")
    axs[n, 0].set_title("perception error of the clearance (off-scale = nearest obstacle not detected)", fontsize=9)
    axs[n, 0].legend(fontsize=7, ncol=3)
    for i, (ax, r) in enumerate(zip(axs[:n, 0], results)):
        lg = r["log"]
        ax.plot(lg[:, 0], lg[:, 6], "k-", lw=1.6, label="true clearance $d$")
        pc = np.where(np.isfinite(lg[:, 7]), lg[:, 7], np.nan)
        ax.plot(lg[:, 0], pc, ".", color=_color(r["name"], i), ms=3, label=r"perceived $\hat d$")
        ax.axhline(rr, color="r", ls="--", lw=1, label="collision ($d$ = robot radius)")
        ax.set_ylim(0, 6)
        ax.set_ylabel("distance [m]")
        ax.set_title(f"{r['name']}: alpha = {r['cfg'].extinction_per_m:.3f} 1/m", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    axs[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_detection(results, nav, path):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cfg0 = results[0]["cfg"]
    alphas = np.logspace(-4, 0, 60)
    for rho, ls_ in ((0.05, ":"), (0.1, "--"), (0.3, "-"), (0.85, "-.")):
        R = [ls.max_range_estimate(ls.replace(cfg0, air_extinction_per_m=a, rain_mm_per_h=0.0), rho) for a in alphas]
        a1.loglog(alphas, R, "k" + ls_, lw=1.2, label=fr"$\rho$ = {rho}")
    for i, r in enumerate(results):
        a1.axvline(r["cfg"].extinction_per_m, color=_color(r["name"], i), lw=2, alpha=0.7, label=r["name"])
    a1.set_xlabel(r"extinction coefficient $\alpha$ [1/m]")
    a1.set_ylabel(r"max. detection range $R_{max}$ [m]")
    a1.set_title("FOV radius from the LiDAR equation (normal incidence)")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=7)
    labels = [k for k in results[0]["first_det"]]
    xs = np.arange(len(labels))
    wd = 0.8 / len(results)
    for i, r in enumerate(results):
        vals = [r["first_det"][k] for k in labels]
        a2.bar(xs + i * wd, np.nan_to_num(vals, nan=0.0), wd, color=_color(r["name"], i), label=r["name"])
    a2.set_xticks(xs + 0.4 - wd / 2)
    a2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    a2.set_ylabel("distance at first detection [m]")
    a2.set_title("Range at which each obstacle is first perceived (0 = never)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def animate(res, nav, path, every, fps):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    lg, fr, rb = res["log"], res["frames"], nav["robot"]
    idx = list(range(0, len(fr), every))
    rmax = ls.max_range_estimate(res["cfg"], 0.1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})

    def draw(j):
        k = idx[j]
        ax.clear()
        ax2.clear()
        draw_scene(ax, res["scene"], nav)
        x, y, th = lg[k, 1:4]
        f = fr[k]
        ax.plot(f["gnd"][:, 0], f["gnd"][:, 1], ".", color="#9ecae1", ms=1, zorder=1)
        ax.plot(f["obs"][:, 0], f["obs"][:, 1], ".", color="#e6550d", ms=2, zorder=6)
        ends = np.stack([x + f["L"] * np.cos(res["H"]), y + f["L"] * np.sin(res["H"])], 1)
        for e in ends:
            ax.plot([x, e[0]], [y, e[1]], "-", color="#31a354", lw=0.5, alpha=0.5, zorder=2)
        ax.add_patch(plt.Circle((x, y), min(rmax, 60), fill=False, ls=":", color="purple", zorder=2))
        ax.add_patch(plt.Circle((x, y), rb["radius_m"], color="k", zorder=7))
        ax.arrow(x, y, 0.9 * np.cos(th), 0.9 * np.sin(th), width=0.06, color="y", zorder=8)
        ax.arrow(x, y, 1.6 * np.cos(f["psi"]), 1.6 * np.sin(f["psi"]), width=0.03, color="g", zorder=8)
        ax.plot(lg[:k + 1, 1], lg[:k + 1, 2], "k-", lw=1, zorder=5)
        ax.set_xlim(-5, 32)
        ax.set_ylim(-9, 9)
        ax.set_title(f"{res['name']}   t = {lg[k, 0]:.1f} s   v = {lg[k, 4]:.2f} m/s   "
                     f"returns = {int(lg[k, 8])}  (dotted: R_max for rho = 0.1)", fontsize=9)
        ax2.plot(lg[:k + 1, 0], lg[:k + 1, 6], "k-", label="true $d$")
        ax2.plot(lg[:k + 1, 0], lg[:k + 1, 7], ".", color="#e6550d", ms=3, label=r"perceived $\hat d$")
        ax2.axhline(rb["radius_m"], color="r", ls="--", lw=1)
        ax2.set_xlim(0, lg[-1, 0] + 0.1)
        ax2.set_ylim(0, 6)
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("clearance [m]")
        ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(idx))
    anim.save(path, writer=PillowWriter(fps=fps))
    draw(int(0.47 * (len(idx) - 1)))                      # still frame for the report
    fig.savefig(path.replace(".gif", "_snapshot.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "nav_config.jsonc"))
    ap.add_argument("--scenario", default=None, help="run only this scenario")
    ap.add_argument("--no-gif", action="store_true", help="skip the GIF animation")
    a = ap.parse_args(argv)
    nav = ls.load_config_file(a.config)
    scens = [s for s in nav["scenarios"] if a.scenario in (None, s["name"])]
    if not scens:
        sys.exit(f"no scenario named {a.scenario!r}; available: {[s['name'] for s in nav['scenarios']]}")
    out = os.path.join(os.path.dirname(os.path.abspath(a.config)), nav["output"]["directory"])
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    results = [run(nav, s) for s in scens]
    print(f"simulation time {time.time() - t0:.1f} s")

    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        labels = list(results[0]["first_det"])
        wr.writerow(["scenario", "alpha_per_m", "Rmax_rho0.05", "Rmax_rho0.3", "status", "time_s", "path_m",
                     "min_true_clearance_m", "min_perceived_clearance_m", "mean_returns"]
                    + [f"first_det_{k}" for k in labels])
        for r in results:
            lg = r["log"]
            wr.writerow([r["name"], f"{r['cfg'].extinction_per_m:.4f}",
                         f"{ls.max_range_estimate(r['cfg'], 0.05):.1f}", f"{ls.max_range_estimate(r['cfg'], 0.3):.1f}",
                         r["status"], f"{lg[-1, 0]:.1f}",
                         f"{np.sum(np.linalg.norm(np.diff(lg[:, 1:3], axis=0), axis=1)):.2f}",
                         f"{lg[:, 6].min():.3f}", f"{np.nanmin(np.where(np.isfinite(lg[:, 7]), lg[:, 7], np.nan)):.3f}",
                         f"{lg[:, 8].mean():.0f}"] + [f"{r['first_det'][k]:.2f}" for k in labels])
        for r in results:
            np.savetxt(os.path.join(out, f"log_{r['name']}.csv"), r["log"], delimiter=",", fmt="%.4f",
                       header="t,x,y,theta,v,omega,true_clearance,perceived_clearance,n_returns,n_obstacle_pts,dist_goal",
                       comments="")

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed -> no figures (pip install matplotlib)")
        return 0
    plot_trajectories(results, nav, os.path.join(out, "trajectories.png"))
    plot_clearance(results, nav, os.path.join(out, "clearance.png"))
    plot_detection(results, nav, os.path.join(out, "detection_range.png"))
    if not a.no_gif:
        for r in results:
            if r["name"] in nav["output"].get("gif_scenarios", []):
                try:
                    animate(r, nav, os.path.join(out, f"nav_{r['name']}.gif"),
                            nav["output"].get("gif_every", 2), nav["output"].get("gif_fps", 8))
                except Exception as e:  # pillow missing, etc.
                    print(f"GIF for {r['name']} skipped: {e} (pip install pillow)")
    print(f"outputs written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
