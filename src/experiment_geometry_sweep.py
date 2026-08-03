"""Camera geometry sweep: which viewing angles support which research
questions?

The camera orbits the scene: azimuth alpha is the horizontal angle between
the viewing direction and the flight direction (alpha=90 side-on,
alpha~10 nearly behind the kicker), at two horizontal distances and two
heights covering typical broadcast positions. Focal length scales with
distance so the image scale at the scene stays fixed (~43 px/m, matching
the Phase 0 camera) — geometry is varied, resolution is not.

50 realizations per config at 2 px, best constraint config (0.3 m crossing
box measured with 0.2 m error). Reports 68% width and bias for speed,
elevation, azimuth, w_perp against alpha.

Run from repo root:  python -m src.experiment_geometry_sweep
Saves reports/phase0_geometry_sweep.png and .npz.
"""

import os
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.synthetic import generate_track, look_at_camera
from src.fitting import fit_flight, flight_quantities
from src.validate_recovery import (P0_TRUE, V0_TRUE, OMEGA_TRUE, FPS,
                                   CROSS_MEAS_SIGMA, BOX_HALF,
                                   goal_plane_crossing)

ALPHAS = [90.0, 60.0, 40.0, 25.0, 10.0]
DISTANCES = [40.0, 80.0]     # horizontal, m
HEIGHTS = [10.0, 22.0]       # m
N_REAL = 50
NOISE_PX = 2.0
PX_PER_M = 43.0              # image scale at the scene, held fixed

TARGET = np.array([0.0, 11.0, 1.5])
GOAL_X, GOAL_Z = goal_plane_crossing(P0_TRUE, V0_TRUE, OMEGA_TRUE)
TRUTH_Q = flight_quantities(np.concatenate([V0_TRUE, OMEGA_TRUE]))
REPORT_ROWS = [0, 1, 2, 3]   # speed, elevation, azimuth, w_perp

CONFIGS = [(a, d, h) for a in ALPHAS for d in DISTANCES for h in HEIGHTS]


def camera_for(alpha_deg, dist, height):
    """Camera on the main-stand side, viewing direction alpha degrees off
    the horizontal flight direction."""
    f = V0_TRUE[:2] / np.linalg.norm(V0_TRUE[:2])
    a = np.radians(alpha_deg)
    view = np.array([np.cos(a) * f[0] - np.sin(a) * f[1],
                     np.sin(a) * f[0] + np.cos(a) * f[1]])
    pos = np.array([TARGET[0] - dist * view[0],
                    TARGET[1] - dist * view[1], height])
    focal = PX_PER_M * np.linalg.norm(pos - TARGET)
    return look_at_camera(pos, TARGET, focal)


def run_one(job):
    cfg_i, seed = job
    alpha, dist, height = CONFIGS[cfg_i]
    camera = camera_for(alpha, dist, height)
    track = generate_track(P0_TRUE, V0_TRUE, OMEGA_TRUE, camera,
                           fps=FPS, noise_px=0.0, seed=0)
    rng = np.random.default_rng(400000 + cfg_i * 1000 + seed)
    box = (GOAL_X + rng.normal(0, CROSS_MEAS_SIGMA),
           GOAL_Z + rng.normal(0, CROSS_MEAS_SIGMA), BOX_HALF)
    uv = track["uv_true"] + rng.normal(0.0, NOISE_PX, size=track["uv_true"].shape)
    theta, _, res = fit_flight(track["times"], uv, camera, box=box,
                               noise_px=NOISE_PX)
    if res.status <= 0:
        return None
    return cfg_i, flight_quantities(theta)


def main():
    jobs = [(i, s) for i in range(len(CONFIGS)) for s in range(N_REAL)]
    print(f"geometry sweep: {len(CONFIGS)} configs x {N_REAL} fits...")
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        rows = [r for r in pool.map(run_one, jobs) if r is not None]

    widths = np.full((len(CONFIGS), 4), np.nan)
    biases = np.full_like(widths, np.nan)
    counts = np.zeros(len(CONFIGS), dtype=int)
    for i in range(len(CONFIGS)):
        q = np.array([r[1] for r in rows if r[0] == i])
        counts[i] = len(q)
        if len(q) < 10:
            continue
        lo, hi = np.percentile(q[:, REPORT_ROWS], [16, 84], axis=0)
        widths[i] = hi - lo
        biases[i] = np.median(q[:, REPORT_ROWS], axis=0) - TRUTH_Q[REPORT_ROWS]

    names = ["speed [m/s]", "elev [deg]", "azim [deg]", "w_perp [rpm]"]
    print(f"\n{'alpha':>6}{'dist':>6}{'ht':>5}"
          + "".join(f"{n + ' w/b':>22}" for n in names))
    for i, (a, d, h) in enumerate(CONFIGS):
        print(f"{a:>6.0f}{d:>6.0f}{h:>5.0f}"
              + "".join(f"{widths[i, j]:>12.2f}/{biases[i, j]:>8.2f}"
                        for j in range(4))
              + f"   ({counts[i]}/{N_REAL})")

    os.makedirs("reports", exist_ok=True)
    np.savez("reports/geometry_sweep.npz", configs=CONFIGS, widths=widths,
             biases=biases, counts=counts)
    plot(widths, biases, names)


def plot(widths, biases, names):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    styles = {(40, 10): "o-", (40, 22): "s--", (80, 10): "^-", (80, 22): "v--"}
    for j in range(4):
        ax = axes[j]
        for (d, h), st in styles.items():
            sel = [i for i, (a, dd, hh) in enumerate(CONFIGS)
                   if dd == d and hh == h]
            ax.plot([CONFIGS[i][0] for i in sel], widths[sel, j], st,
                    label=f"d={d:.0f} h={h:.0f}")
        ax.set_xlabel("viewing angle alpha [deg]")
        ax.set_title(f"{names[j]} 68% width")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    ax = axes[4]
    for (d, h), st in styles.items():
        sel = [i for i, (a, dd, hh) in enumerate(CONFIGS)
               if dd == d and hh == h]
        ax.plot([CONFIGS[i][0] for i in sel], biases[sel, 3], st,
                label=f"d={d:.0f} h={h:.0f}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("viewing angle alpha [deg]")
    ax.set_title("w_perp bias [rpm]")
    ax.grid(alpha=0.3)
    fig.suptitle("camera geometry sweep, 2 px noise, 0.3 m crossing box "
                 "(alpha=90 side-on, alpha=10 behind kicker)")
    fig.tight_layout()
    out = "reports/phase0_geometry_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
