"""Phase 0 noise sweep on the production fitting model: how do interval
widths, spin shrinkage, and interval calibration move with pixel noise?

For each noise level (0.5 - 5 px), 80 realizations of the standard 23 m
curler: noisy track + goal-plane crossing measured with 0.2 m error
(0.3 m box), fit with fit_flight, intervals from bootstrap_flight
(15 refits). Reports per level:
  - 68% width of the point-estimate scatter per flight quantity
  - median bias per quantity (spin shrinkage shows up here)
  - interval coverage of truth, raw percentile vs spin-inflated

Run from repo root:  python -m src.experiment_noise_sweep
Saves reports/phase0_noise_sweep.png and .npz.
"""

import os
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.synthetic import broadcast_camera, generate_track
from src.fitting import (QUANTITY_NAMES, SPIN_ROWS, SPIN_INTERVAL_INFLATION,
                         bootstrap_flight, fit_flight, flight_quantities)
from src.validate_recovery import (P0_TRUE, V0_TRUE, OMEGA_TRUE, FPS,
                                   CROSS_MEAS_SIGMA, BOX_HALF,
                                   goal_plane_crossing)

NOISE_LEVELS = [0.5, 1.0, 2.0, 3.0, 5.0]
N_OUTER = 80
N_INNER = 15

CAMERA = broadcast_camera()
TRACK = generate_track(P0_TRUE, V0_TRUE, OMEGA_TRUE, CAMERA, fps=FPS,
                       noise_px=0.0, seed=0)
GOAL_X, GOAL_Z = goal_plane_crossing(P0_TRUE, V0_TRUE, OMEGA_TRUE)
TRUTH_Q = flight_quantities(np.concatenate([V0_TRUE, OMEGA_TRUE]))


def run_one(job):
    """One realization: noisy track + measured box -> fit -> intervals."""
    level_i, seed = job
    noise_px = NOISE_LEVELS[level_i]
    rng = np.random.default_rng(50000 + level_i * 1000 + seed)
    box = (GOAL_X + rng.normal(0, CROSS_MEAS_SIGMA),
           GOAL_Z + rng.normal(0, CROSS_MEAS_SIGMA), BOX_HALF)
    uv = TRACK["uv_true"] + rng.normal(0.0, noise_px, size=TRACK["uv_true"].shape)

    theta, _, res = fit_flight(TRACK["times"], uv, CAMERA, box=box,
                               noise_px=noise_px)
    if res.status <= 0:
        return level_i, None, None, None
    intervals, samples = bootstrap_flight(
        TRACK["times"], uv, CAMERA, theta, box=box, noise_px=noise_px,
        n_boot=N_INNER, seed=90000 + level_i * 1000 + seed)

    # raw percentile intervals (un-inflated), for the calibration comparison
    q_samples = np.array([flight_quantities(th) for th, _ in samples])
    lo, hi = np.percentile(q_samples, [16, 84], axis=0)
    hit_raw = (TRUTH_Q >= lo) & (TRUTH_Q <= hi)
    hit_infl = (TRUTH_Q >= intervals[:, 0]) & (TRUTH_Q <= intervals[:, 1])
    return level_i, flight_quantities(theta), hit_raw, hit_infl


def main():
    jobs = [(i, s) for i in range(len(NOISE_LEVELS)) for s in range(N_OUTER)]
    print(f"noise sweep: {len(jobs)} realizations x {1 + N_INNER} fits...")
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        results = pool.map(run_one, jobs)

    widths = np.full((len(NOISE_LEVELS), 5), np.nan)
    biases = np.full_like(widths, np.nan)
    cov_raw = np.full_like(widths, np.nan)
    cov_infl = np.full_like(widths, np.nan)
    for i, noise in enumerate(NOISE_LEVELS):
        rows = [r for r in results if r[0] == i and r[1] is not None]
        q = np.array([r[1] for r in rows])
        lo, hi = np.percentile(q, [16, 84], axis=0)
        widths[i], biases[i] = hi - lo, np.median(q, axis=0) - TRUTH_Q
        cov_raw[i] = np.mean([r[2] for r in rows], axis=0)
        cov_infl[i] = np.mean([r[3] for r in rows], axis=0)

        print(f"\nnoise {noise:.1f} px ({len(rows)}/{N_OUTER} converged):")
        print(f"  {'quantity':<18}{'width':>9}{'bias':>9}"
              f"{'cov raw':>9}{'cov infl':>10}")
        for j, name in enumerate(QUANTITY_NAMES):
            print(f"  {name:<18}{widths[i, j]:>9.2f}{biases[i, j]:>9.2f}"
                  f"{100*cov_raw[i, j]:>8.0f}%{100*cov_infl[i, j]:>9.0f}%")

    os.makedirs("reports", exist_ok=True)
    np.savez("reports/noise_sweep.npz", noise=NOISE_LEVELS, widths=widths,
             biases=biases, cov_raw=cov_raw, cov_infl=cov_infl)
    plot(widths, biases, cov_raw, cov_infl)


def plot(widths, biases, cov_raw, cov_infl):
    noise = np.array(NOISE_LEVELS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for j in [0, 1, 2, 3]:
        ax.loglog(noise, 100 * widths[:, j] / abs(TRUTH_Q[j]), "o-",
                  label=QUANTITY_NAMES[j])
    ax.set_xlabel("pixel noise [px]")
    ax.set_ylabel("68% width, % of true value")
    ax.set_title("precision vs noise")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(noise, TRUTH_Q[3] + biases[:, 3], yerr=widths[:, 3] / 2,
                fmt="o-", capsize=3, label="median recovered $\\omega_\\perp$")
    ax.axhline(TRUTH_Q[3], color="k", ls="--", lw=1, label="truth")
    ax.set_xlabel("pixel noise [px]")
    ax.set_ylabel("$\\omega_\\perp$ [rpm]")
    ax.set_title("spin shrinkage vs noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for j, style in [(0, "s-"), (3, "o-")]:
        ax.plot(noise, 100 * cov_raw[:, j], style, mfc="none",
                label=f"{QUANTITY_NAMES[j]} raw")
    ax.plot(noise, 100 * cov_infl[:, 3], "o-",
            label=f"w_perp x{SPIN_INTERVAL_INFLATION}")
    ax.axhline(68, color="k", ls="--", lw=1)
    ax.set_xlabel("pixel noise [px]")
    ax.set_ylabel("68% interval coverage of truth [%]")
    ax.set_ylim(0, 100)
    ax.set_title("interval calibration vs noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = "reports/phase0_noise_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
