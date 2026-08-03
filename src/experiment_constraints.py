"""Constraint experiment: how much of the depth degeneracy comes from the
free launch point, and does an endpoint constraint buy anything beyond that?

Four configurations, same 23 m curler as validate_recovery, 50 noise
realizations each at 2 px:
  A baseline      - all 9 parameters free (current setup)
  B z0 pinned     - z0 fixed at ball radius 0.11 m (ball sits on the grass)
  C launch prior  - B + Gaussian prior sigma=0.2 m on x0, y0, centred on
                    truth (optimistic stand-in for a homography fix of the
                    ball's spot on the pitch plane)
  D endpoint      - C + goal-plane crossing within +/-0.3 m of the true
                    crossing (hinge penalty, zero inside the tolerance box)

Spin is reported decomposed about the instantaneous launch velocity
direction v_hat = v0/|v0|: w_par = w . v_hat (signed; produces no Magnus
force, so it is structurally unobservable from the trajectory) and
w_perp = |w - w_par*v_hat| (transverse; produces all the Magnus force).

Run from repo root:  python -m src.experiment_constraints
Prints 68% interval widths per config, saves reports/phase0_constraints.png.
"""

import os
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from src.physics import simulate
from src.synthetic import GOAL_HALF_WIDTH, broadcast_camera, generate_track
from src.fitting import LOWER, UPPER, residuals, default_guess, unpack
from src.validate_recovery import (P0_TRUE, V0_TRUE, OMEGA_TRUE, FPS, NOISE_PX,
                                   RAD_TO_RPM, goal_plane_crossing)

N_REALIZATIONS = 50
Z0_BALL = 0.11
PRIOR_SIGMA_XY = 0.2      # m, homography-level launch point knowledge
ENDPOINT_TOL = 0.3        # m, allowed miss vs the true goal-plane crossing
ENDPOINT_WEIGHT = 0.05    # m, stiffness of the hinge once outside the box

CONFIGS = ["A baseline", "B z0 pinned", "C launch prior", "D endpoint"]

CAMERA = broadcast_camera()
TRACK = generate_track(P0_TRUE, V0_TRUE, OMEGA_TRUE, CAMERA, fps=FPS,
                       noise_px=0.0, seed=0)
GOAL_X, GOAL_Z = goal_plane_crossing(P0_TRUE, V0_TRUE, OMEGA_TRUE)

# 8-parameter versions (z0 removed) for configs B-D
IDX8 = [0, 1, 3, 4, 5, 6, 7, 8]
LOWER8, UPPER8 = LOWER[IDX8], UPPER[IDX8]


def to_theta9(theta8):
    return np.insert(theta8, 2, Z0_BALL)


def crossing_penalty(theta9):
    """Hinge residuals on the goal-plane crossing, zero inside the box."""
    p0, v0, omega = unpack(theta9)
    times = np.linspace(0.0, 1.8, 180)
    xyz = simulate(p0, v0, omega, times)
    below = xyz[:, 1] <= 0.0
    if not below.any():
        return np.array([20.0, 20.0])  # never reaches the goal plane
    i = np.argmax(below)
    a, b = xyz[i - 1], xyz[i]
    cross = a + (a[1] / (a[1] - b[1])) * (b - a)
    ex = max(0.0, abs(cross[0] - GOAL_X) - ENDPOINT_TOL)
    ez = max(0.0, abs(cross[2] - GOAL_Z) - ENDPOINT_TOL)
    return np.array([ex, ez]) / ENDPOINT_WEIGHT


def make_residual(config, uv_obs):
    """Residual function for one config. Pixel residuals are whitened by
    NOISE_PX so prior and penalty terms are in consistent sigma units."""
    times = TRACK["times"]

    def f(theta):
        theta9 = theta if config == "A baseline" else to_theta9(theta)
        r = [residuals(theta9, times, uv_obs, CAMERA) / NOISE_PX]
        if config in ("C launch prior", "D endpoint"):
            r.append((theta9[:2] - P0_TRUE[:2]) / PRIOR_SIGMA_XY)
        if config == "D endpoint":
            r.append(crossing_penalty(theta9))
        return np.concatenate(r)

    return f


def fit_one(job):
    config, seed = job
    rng = np.random.default_rng(2000 + seed)
    uv = TRACK["uv_true"] + rng.normal(0.0, NOISE_PX, size=TRACK["uv_true"].shape)
    if config == "A baseline":
        guess, bounds = default_guess(), (LOWER, UPPER)
    else:
        guess, bounds = default_guess()[IDX8], (LOWER8, UPPER8)
    res = least_squares(make_residual(config, uv), guess, bounds=bounds,
                        method="trf", x_scale="jac")
    theta9 = res.x if config == "A baseline" else to_theta9(res.x)
    return config, seed, theta9, int(res.status)


def metrics(theta9):
    """speed, elevation, azimuth, |w_perp|, w_par (signed), axis error [deg].

    Spin is decomposed about the fit's own instantaneous launch velocity
    direction (v at t=0), the axis that determines what Magnus can see.
    """
    p0, v0, omega = unpack(theta9)
    v_hat = v0 / np.linalg.norm(v0)
    w_par = omega @ v_hat
    w_perp = np.linalg.norm(omega - w_par * v_hat)
    axis_true = OMEGA_TRUE / np.linalg.norm(OMEGA_TRUE)
    axis = omega / np.linalg.norm(omega)
    axis_err = np.degrees(np.arccos(np.clip(axis_true @ axis, -1, 1)))
    return np.array([
        np.linalg.norm(v0),
        np.degrees(np.arctan2(v0[2], np.hypot(v0[0], v0[1]))),
        np.degrees(np.arctan2(v0[0], -v0[1])),
        w_perp * RAD_TO_RPM,
        w_par * RAD_TO_RPM,
        axis_err,
    ])


METRIC_NAMES = ["speed [m/s]", "elev [deg]", "azim [deg]",
                "|w_perp| [rpm]", "w_par [rpm]", "axis err [deg]"]


def main():
    truth9 = np.concatenate([P0_TRUE, V0_TRUE, OMEGA_TRUE])
    m_true = metrics(truth9)
    print("truth:", "  ".join(f"{n}={v:.1f}" for n, v in zip(METRIC_NAMES, m_true)))
    print(f"goal-plane crossing (truth): x={GOAL_X:+.2f}, z={GOAL_Z:.2f}\n")

    jobs = [(c, s) for c in CONFIGS for s in range(N_REALIZATIONS)]
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        results = pool.map(fit_one, jobs)

    fits = {c: [] for c in CONFIGS}
    for config, seed, theta9, status in results:
        if status > 0:
            fits[config].append(theta9)

    print(f"68% interval WIDTHS (84th - 16th pct) over {N_REALIZATIONS} "
          f"noise realizations:\n")
    header = f"{'config':<16}" + "".join(f"{n:>16}" for n in METRIC_NAMES)
    print(header)
    for config in CONFIGS:
        m = np.array([metrics(t) for t in fits[config]])
        lo, hi = np.percentile(m, [16, 84], axis=0)
        row = f"{config:<16}" + "".join(f"{w:>16.2f}" for w in hi - lo)
        print(row + f"   ({len(fits[config])}/{N_REALIZATIONS} converged)")

    print("\nlaunch-point scatter, median |p0_fit - p0_true| [m]:")
    for config in CONFIGS:
        d = [np.linalg.norm(unpack(t)[0] - P0_TRUE) for t in fits[config]]
        print(f"  {config:<16}{np.median(d):.2f}")

    plot(fits)


def plot(fits):
    times_fine = np.linspace(0.0, TRACK["times"][-1], 120)
    xyz_true = simulate(P0_TRUE, V0_TRUE, OMEGA_TRUE, times_fine)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)
    for ax, config in zip(axes, CONFIGS):
        for t in fits[config]:
            xyz = simulate(*unpack(t), times_fine)
            ax.plot(xyz[:, 0], xyz[:, 1], color="0.75", lw=0.7, zorder=1)
        ax.plot(xyz_true[:, 0], xyz_true[:, 1], c="tab:blue", lw=2,
                label="true", zorder=3)
        ax.plot([-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [0, 0], "k-", lw=3)
        ax.set_title(config)
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [m]")
    axes[0].legend()
    fig.suptitle("top-down trajectory fans, 50 noise realizations at 2 px")
    fig.tight_layout()
    os.makedirs("reports", exist_ok=True)
    out = "reports/phase0_constraints.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
