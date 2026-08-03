"""Residual-dependent spin correction: replace the constant 1.5x inflation
with a bias offset b(rms) and inflation k(rms) looked up from each clip's
own measured reprojection RMS.

Modes:
  calibrate — 5 noise levels x 80 realizations, per-fit records (measured
              rms, point estimate, raw percentile interval) -> npz, then
              solve per level: b = median bias of w_perp, k = smallest
              inflation reaching 68% coverage after recentring by b.
              The printed table goes into fitting.py.
  verify    — OFF-GRID noise levels (0.75/1.5/2.5/4 px) x 60 fresh seeds,
              through the shipped bootstrap_flight (which applies the
              correction from each fit's own rms). Coverage should hold
              across the whole range if the interpolation is sound.

Run from repo root:  python -m src.experiment_spin_calibration <mode>
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

from src.synthetic import broadcast_camera, generate_track
from src.fitting import (QUANTITY_NAMES, bootstrap_flight, fit_flight,
                         flight_quantities, reprojection_rms)
from src.validate_recovery import (P0_TRUE, V0_TRUE, OMEGA_TRUE, FPS,
                                   CROSS_MEAS_SIGMA, BOX_HALF,
                                   goal_plane_crossing)

CAL_LEVELS = [0.5, 1.0, 2.0, 3.0, 5.0]
VER_LEVELS = [0.75, 1.5, 2.5, 4.0]
N_CAL, N_VER, N_INNER = 80, 60, 15

CAMERA = broadcast_camera()
TRACK = generate_track(P0_TRUE, V0_TRUE, OMEGA_TRUE, CAMERA, fps=FPS,
                       noise_px=0.0, seed=0)
GOAL_X, GOAL_Z = goal_plane_crossing(P0_TRUE, V0_TRUE, OMEGA_TRUE)
TRUTH_Q = flight_quantities(np.concatenate([V0_TRUE, OMEGA_TRUE]))


def observe_and_fit(noise_px, rng):
    box = (GOAL_X + rng.normal(0, CROSS_MEAS_SIGMA),
           GOAL_Z + rng.normal(0, CROSS_MEAS_SIGMA), BOX_HALF)
    uv = TRACK["uv_true"] + rng.normal(0.0, noise_px, size=TRACK["uv_true"].shape)
    theta, p0, res = fit_flight(TRACK["times"], uv, CAMERA, box=box,
                                noise_px=noise_px)
    return theta, p0, res, uv, box


def cal_job(job):
    """Record measured rms, w_perp estimate, and the RAW percentile
    interval, so any (b, k) rule can be evaluated offline."""
    level_i, seed = job
    rng = np.random.default_rng(200000 + level_i * 1000 + seed)
    theta, p0, res, uv, box = observe_and_fit(CAL_LEVELS[level_i], rng)
    if res.status <= 0:
        return None
    rms = reprojection_rms(theta, p0, TRACK["times"], uv, CAMERA)
    _, samples = bootstrap_flight(TRACK["times"], uv, CAMERA, theta, box=box,
                                  noise_px=CAL_LEVELS[level_i],
                                  n_boot=N_INNER,
                                  seed=210000 + level_i * 1000 + seed)
    q = np.array([flight_quantities(th) for th, _ in samples])
    lo, hi = np.percentile(q[:, 3], [16, 84])
    return level_i, rms, flight_quantities(theta)[3], lo, hi


def calibrate():
    jobs = [(i, s) for i in range(len(CAL_LEVELS)) for s in range(N_CAL)]
    print(f"calibration: {len(jobs)} realizations x {1 + N_INNER} fits...")
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        rows = [r for r in pool.map(cal_job, jobs) if r is not None]
    rows = np.array(rows)
    np.savez("reports/spin_calibration.npz", rows=rows)

    print(f"\n{'noise':>6}{'med rms':>9}{'bias b':>9}{'k':>6}{'cov':>6}  (truth w_perp {TRUTH_Q[3]:.0f} rpm)")
    print("calibration table for fitting.py (rms, bias, inflation):")
    for i, noise in enumerate(CAL_LEVELS):
        r = rows[rows[:, 0] == i]
        rms_med = np.median(r[:, 1])
        b = np.median(r[:, 2]) - TRUTH_Q[3]
        centre, half = 0.5 * (r[:, 3] + r[:, 4]), 0.5 * (r[:, 4] - r[:, 3])
        for k in np.arange(1.0, 3.01, 0.05):
            hit = (np.abs(TRUTH_Q[3] - (centre - b)) <= k * half)
            if hit.mean() >= 0.68:
                break
        print(f"{noise:>6.1f}{rms_med:>9.2f}{b:>9.1f}{k:>6.2f}{100*hit.mean():>5.0f}%")


def ver_job(job):
    """Through the SHIPPED code path: bootstrap_flight applies the
    rms-dependent correction internally."""
    level_i, seed = job
    noise_px = VER_LEVELS[level_i]
    rng = np.random.default_rng(300000 + level_i * 1000 + seed)
    theta, p0, res, uv, box = observe_and_fit(noise_px, rng)
    if res.status <= 0:
        return None
    intervals, _ = bootstrap_flight(TRACK["times"], uv, CAMERA, theta, box=box,
                                    noise_px=noise_px, n_boot=N_INNER,
                                    seed=310000 + level_i * 1000 + seed)
    hit = (TRUTH_Q >= intervals[:, 0]) & (TRUTH_Q <= intervals[:, 1])
    return level_i, *hit


def verify():
    jobs = [(i, s) for i in range(len(VER_LEVELS)) for s in range(N_VER)]
    print(f"verification (off-grid levels): {len(jobs)} realizations...")
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        rows = [r for r in pool.map(ver_job, jobs) if r is not None]
    rows = np.array(rows)
    print(f"\ncoverage through shipped intervals (nominal 68%):")
    print(f"{'noise':>6}" + "".join(f"{n:>16}" for n in QUANTITY_NAMES))
    for i, noise in enumerate(VER_LEVELS):
        r = rows[rows[:, 0] == i][:, 1:]
        print(f"{noise:>6.2f}" + "".join(f"{100*c:>15.0f}%" for c in r.mean(axis=0))
              + f"   ({len(r)}/{N_VER})")


if __name__ == "__main__":
    {"calibrate": calibrate, "verify": verify}[sys.argv[1]]()
