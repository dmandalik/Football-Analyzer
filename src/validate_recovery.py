"""Phase 0 recovery test on the production fitting model:
z0 pinned, launch point from the first-frame ray, goal-plane crossing as a
measured hinge box, percentile bootstrap with calibrated spin inflation.

Run from the repo root:  python -m src.validate_recovery

Steps:
  1. noiseless fit  — proves the machinery is exact
  2. noisy fit      — 2 px noise, crossing measured with 0.2 m error
  3. bootstrap      — the intervals the pipeline ships (19 refits;
     spin rows inflated by the synthetic-calibrated factor)

Prints recovered vs. true values and saves reports/phase0_recovery.png.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.physics import simulate
from src.synthetic import GOAL_HALF_WIDTH, GOAL_HEIGHT, broadcast_camera, generate_track
from src.fitting import (RAD_TO_RPM, QUANTITY_NAMES, bootstrap_flight,
                         fit_flight, flight_quantities, reprojection_rms,
                         spin_correction)

# ground truth: right-footed curler from the left side, ~25 m/s, heavy sidespin
P0_TRUE = np.array([-7.0, 23.0, 0.11])
V0_TRUE = np.array([9.0, -22.5, 6.8])
OMEGA_TRUE = np.array([-10.0, 6.0, -62.0])

FPS = 25.0
NOISE_PX = 2.0
SEED = 7
CROSS_MEAS_SIGMA = 0.2  # m, error on the measured goal-plane crossing
BOX_HALF = 0.3


def goal_plane_crossing(p0, v0, omega):
    """(x, z) where the trajectory crosses y=0, by dense simulation."""
    times = np.linspace(0.0, 2.0, 1001)
    xyz = simulate(p0, v0, omega, times)
    idx = np.argmax(xyz[:, 1] <= 0.0)
    a, b = xyz[idx - 1], xyz[idx]
    frac = a[1] / (a[1] - b[1])
    cross = a + frac * (b - a)
    return cross[0], cross[2]


def main():
    truth6 = np.concatenate([V0_TRUE, OMEGA_TRUE])
    camera = broadcast_camera()

    gx, gz = goal_plane_crossing(P0_TRUE, V0_TRUE, OMEGA_TRUE)
    in_goal = abs(gx) < GOAL_HALF_WIDTH and 0.0 < gz < GOAL_HEIGHT
    print(f"true trajectory crosses goal plane at x={gx:+.2f} m, z={gz:.2f} m "
          f"({'inside' if in_goal else 'OUTSIDE'} the goal mouth)")

    track = generate_track(P0_TRUE, V0_TRUE, OMEGA_TRUE, camera,
                           fps=FPS, noise_px=NOISE_PX, seed=SEED)
    print(f"{len(track['times'])} frames at {FPS:.0f} fps over "
          f"{track['t_end']:.2f} s\n")

    # step 1: noiseless recovery
    th0, p0_0, _ = fit_flight(track["times"], track["uv_true"], camera)
    print(f"[1] noiseless fit: max |v,w error| = "
          f"{np.max(np.abs(th0 - truth6)):.2e}, "
          f"|launch point error| = {np.linalg.norm(p0_0 - P0_TRUE):.2e} m\n")

    # step 2: noisy fit with a measured crossing box
    rng = np.random.default_rng(SEED)
    box = (gx + rng.normal(0, CROSS_MEAS_SIGMA),
           gz + rng.normal(0, CROSS_MEAS_SIGMA), BOX_HALF)
    theta, p0, result = fit_flight(track["times"], track["uv_noisy"], camera,
                                   box=box)
    print(f"[2] noisy fit: converged (status {result.status}, "
          f"nfev {result.nfev}); measured crossing box centre "
          f"({box[0]:+.2f}, {box[1]:.2f}) +/- {BOX_HALF}\n")
    print(f"    launch point: derived ({p0[0]:+.2f}, {p0[1]:.2f}, {p0[2]:.2f}), "
          f"true ({P0_TRUE[0]:+.2f}, {P0_TRUE[1]:.2f}, {P0_TRUE[2]:.2f})")

    # step 3: shipped intervals
    intervals, samples = bootstrap_flight(track["times"], track["uv_noisy"],
                                          camera, theta, box=box, seed=SEED)
    rms = reprojection_rms(theta, p0, track["times"], track["uv_noisy"], camera)
    bias, _ = spin_correction(rms)
    q_true, q_fit = flight_quantities(truth6), flight_quantities(theta)
    q_fit[3] -= bias  # debiased w_perp, matching the interval's recentring
    print(f"\n    measured rms {rms:.2f} px -> w_perp debias {-bias:+.0f} rpm")
    print(f"\n{'quantity':<18}{'true':>10}{'recovered':>12}{'68% interval':>20}")
    for i, name in enumerate(QUANTITY_NAMES):
        note = ("  (debiased)" if i == 3 else
                "  (unobservable)" if i == 4 else "")
        print(f"{name:<18}{q_true[i]:>10.2f}{q_fit[i]:>12.2f}"
              f"{f'[{intervals[i, 0]:.2f}, {intervals[i, 1]:.2f}]':>20}{note}")

    fx, fz = goal_plane_crossing(p0, theta[:3], theta[3:])
    print(f"\nfitted trajectory crosses goal plane at x={fx:+.2f}, z={fz:.2f} "
          f"(true x={gx:+.2f}, z={gz:.2f}) -> miss {np.hypot(fx-gx, fz-gz):.3f} m")

    plot(track, theta, p0, samples)


def plot(track, theta, p0, samples):
    times_fine = np.linspace(0.0, track["times"][-1], 200)
    xyz_true = simulate(P0_TRUE, V0_TRUE, OMEGA_TRUE, times_fine)
    xyz_fit = simulate(p0, theta[:3], theta[3:], times_fine)

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(*xyz_true.T, label="true", lw=2)
    ax.plot(*xyz_fit.T, "--", label="recovered", lw=2)
    gw, gh = GOAL_HALF_WIDTH, GOAL_HEIGHT
    ax.plot([-gw, -gw, gw, gw], [0, 0, 0, 0], [0, gh, gh, 0], "k-", lw=1.5)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    ax.set_title("3D trajectory")
    ax.legend()
    ax.view_init(elev=18, azim=-50)

    ax = fig.add_subplot(1, 3, 2)
    for th_b, p0_b in samples:
        xyz_b = simulate(p0_b, th_b[:3], th_b[3:], times_fine)
        ax.plot(xyz_b[:, 0], xyz_b[:, 1], color="0.8", lw=0.8, zorder=1)
    ax.plot(xyz_true[:, 0], xyz_true[:, 1], lw=2, label="true", zorder=3)
    ax.plot(xyz_fit[:, 0], xyz_fit[:, 1], "--", lw=2, label="recovered", zorder=3)
    ax.plot([-gw, gw], [0, 0], "k-", lw=3)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]")
    ax.set_title("top-down; gray = bootstrap fits")
    ax.axis("equal")
    ax.legend()

    ax = fig.add_subplot(1, 3, 3)
    uv_fit = broadcast_camera().project(xyz_fit)
    ax.scatter(*track["uv_noisy"].T, s=12, c="tab:red", label="observed (2 px noise)")
    ax.plot(*track["uv_true"].T, lw=1, c="tab:blue", label="true projection")
    ax.plot(*uv_fit.T, "--", lw=1.5, c="tab:green", label="fit reprojection")
    ax.invert_yaxis()
    ax.set_xlabel("u [px]"), ax.set_ylabel("v [px]")
    ax.set_title("image plane")
    ax.legend()

    fig.tight_layout()
    os.makedirs("reports", exist_ok=True)
    out = "reports/phase0_recovery.png"
    fig.savefig(out, dpi=130)
    print(f"plot saved to {out}")


if __name__ == "__main__":
    main()
