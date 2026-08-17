"""Spin correction calibrated on ANISOTROPIC tracking noise, keyed on
cross-track residual RMS.

Real SAM 2 tracks show blur-driven anisotropy: along-track noise 3-11x
larger than cross-track (experiment_track_noise). Curvature information
lives cross-track, so the correction is keyed on measured cross-track RMS.
Calibration noise uses along/cross ratio 4 (the fit-grade Ronaldo clip
measures 3.0); a ratio-10 sensitivity check verifies the key is not badly
ratio-dependent.

Modes:
  calibrate — cross-track levels x 80 realizations, records (measured
              rms_cross, w_perp estimate, raw percentile interval) -> npz;
              solves b(rms_cross) and inflation k(rms_cross) per level.
              The printed table goes into fitting.py. Ends with the
              ratio-10 sensitivity block.
  verify    — OFF-GRID cross-track levels, fresh seeds, through the shipped
              bootstrap_flight (anisotropic inner noise matched to the
              fit's own residual decomposition). Coverage should hold.

Run from repo root:  python -m src.experiment_spin_calibration <mode>
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

from src.synthetic import broadcast_camera, generate_track
from src.fitting import (QUANTITY_NAMES, anisotropic_noise, bootstrap_flight,
                         fit_flight, flight_quantities, residual_decomposition)
from src.validate_recovery import (P0_TRUE, V0_TRUE, OMEGA_TRUE, FPS,
                                   CROSS_MEAS_SIGMA, BOX_HALF,
                                   goal_plane_crossing)

RATIO = 4.0                    # along/cross noise ratio for calibration
CAL_LEVELS = [0.5, 1.0, 1.5, 2.0, 3.0]    # cross-track px
VER_LEVELS = [0.75, 1.25, 1.75, 2.5]
R_CHECK = (1.5, 10.0)          # (cross level, ratio) sensitivity spot check
N_CAL, N_VER, N_INNER = 80, 60, 15

CAMERA = broadcast_camera()
TRACK = generate_track(P0_TRUE, V0_TRUE, OMEGA_TRUE, CAMERA, fps=FPS,
                       noise_px=0.0, seed=0)
GOAL_X, GOAL_Z = goal_plane_crossing(P0_TRUE, V0_TRUE, OMEGA_TRUE)
TRUTH_Q = flight_quantities(np.concatenate([V0_TRUE, OMEGA_TRUE]))


def observe_and_fit(sigma_cross, ratio, rng):
    box = (GOAL_X + rng.normal(0, CROSS_MEAS_SIGMA),
           GOAL_Z + rng.normal(0, CROSS_MEAS_SIGMA), BOX_HALF)
    uv = anisotropic_noise(TRACK["uv_true"], sigma_cross,
                           ratio * sigma_cross, rng)
    # scalar whitening; the empirical calibration absorbs the misweighting
    noise_px = sigma_cross * np.sqrt((1 + ratio ** 2) / 2)
    theta, p0, res = fit_flight(TRACK["times"], uv, CAMERA, box=box,
                                noise_px=noise_px)
    return theta, p0, res, uv, box, noise_px


def cal_job(job):
    level_i, seed, sigma_cross, ratio = job
    rng = np.random.default_rng(500000 + level_i * 1000 + seed)
    theta, p0, res, uv, box, noise_px = observe_and_fit(sigma_cross, ratio, rng)
    if res.status <= 0:
        return None
    rms_cross, _ = residual_decomposition(theta, p0, TRACK["times"], uv, CAMERA)
    _, samples = bootstrap_flight(TRACK["times"], uv, CAMERA, theta, box=box,
                                  noise_px=noise_px, n_boot=N_INNER,
                                  seed=510000 + level_i * 1000 + seed)
    q = np.array([flight_quantities(th) for th, _ in samples])
    lo, hi = np.percentile(q[:, 3], [16, 84])
    return level_i, rms_cross, flight_quantities(theta)[3], lo, hi


def solve_level(rows):
    """(median rms_cross, bias b, smallest k reaching 68% after recentring)."""
    rms_med = np.median(rows[:, 1])
    b = np.median(rows[:, 2]) - TRUTH_Q[3]
    centre, half = 0.5 * (rows[:, 3] + rows[:, 4]), 0.5 * (rows[:, 4] - rows[:, 3])
    for k in np.arange(1.0, 3.01, 0.05):
        hit = np.abs(TRUTH_Q[3] - (centre - b)) <= k * half
        if hit.mean() >= 0.68:
            break
    return rms_med, b, k, hit.mean()


def calibrate():
    jobs = [(i, s, lvl, RATIO) for i, lvl in enumerate(CAL_LEVELS)
            for s in range(N_CAL)]
    jobs += [(len(CAL_LEVELS), s, R_CHECK[0], R_CHECK[1]) for s in range(N_CAL)]
    print(f"calibration: {len(jobs)} realizations x {1 + N_INNER} fits "
          f"(ratio {RATIO}, plus ratio-{R_CHECK[1]:.0f} check)...", flush=True)
    rows = []
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        for n, r in enumerate(pool.imap_unordered(cal_job, jobs), 1):
            if r is not None:
                rows.append(r)
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} realizations done", flush=True)
    rows = np.array(rows)
    os.makedirs("reports", exist_ok=True)
    np.savez("reports/spin_calibration_aniso.npz", rows=rows)

    print(f"\ntruth w_perp {TRUTH_Q[3]:.0f} rpm; calibration table for "
          f"fitting.py (rms_cross, bias, inflation):")
    print(f"{'cross px':>9}{'med rms':>9}{'bias b':>9}{'k':>6}{'cov':>6}")
    for i, lvl in enumerate(CAL_LEVELS):
        rms_med, b, k, cov = solve_level(rows[rows[:, 0] == i])
        print(f"{lvl:>9.1f}{rms_med:>9.2f}{b:>9.1f}{k:>6.2f}{100*cov:>5.0f}%")
    rms_med, b, k, cov = solve_level(rows[rows[:, 0] == len(CAL_LEVELS)])
    print(f"\nratio-{R_CHECK[1]:.0f} check at cross {R_CHECK[0]} px: "
          f"med rms {rms_med:.2f}, bias {b:.1f}, k {k:.2f}, cov {100*cov:.0f}% "
          f"(compare to the ratio-{RATIO:.0f} row above)")


def ver_job(job):
    level_i, seed = job
    rng = np.random.default_rng(600000 + level_i * 1000 + seed)
    theta, p0, res, uv, box, noise_px = observe_and_fit(VER_LEVELS[level_i],
                                                        RATIO, rng)
    if res.status <= 0:
        return None
    intervals, _ = bootstrap_flight(TRACK["times"], uv, CAMERA, theta, box=box,
                                    noise_px=noise_px, n_boot=N_INNER,
                                    seed=610000 + level_i * 1000 + seed)
    hit = (TRUTH_Q >= intervals[:, 0]) & (TRUTH_Q <= intervals[:, 1])
    return level_i, *hit


def verify():
    jobs = [(i, s) for i in range(len(VER_LEVELS)) for s in range(N_VER)]
    print(f"verification (off-grid cross levels, ratio {RATIO}): "
          f"{len(jobs)} realizations...", flush=True)
    rows = []
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        for n, r in enumerate(pool.imap_unordered(ver_job, jobs), 1):
            if r is not None:
                rows.append(r)
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} realizations done", flush=True)
    rows = np.array(rows)
    print(f"\ncoverage through shipped intervals (nominal 68%):")
    print(f"{'cross':>6}" + "".join(f"{n:>16}" for n in QUANTITY_NAMES))
    for i, lvl in enumerate(VER_LEVELS):
        r = rows[rows[:, 0] == i][:, 1:]
        print(f"{lvl:>6.2f}" + "".join(f"{100*c:>15.0f}%" for c in r.mean(axis=0))
              + f"   ({len(r)}/{N_VER})")


# NOTE: (1.6, 4.9) was measured on a track later found to be garbage (the
# Ronaldo camera pans; the "flight" was a divot + pan artifact). The
# clipmatch validation therefore characterizes the solver at this
# SYNTHETIC noise point only — it is not validated for any real clip.
# overridable so clip-matched validations can run at any measured noise
# point (workers re-import this module, so use the environment, not argv)
CLIP_POINT = tuple(map(float, os.environ.get(
    "CLIP_POINT", "1.6,4.9").split(",")))
RATIO_RECHECK = {"r4": (1.5, 6.0), "r10": (1.5, 15.0)}
VARIANT_INDEX = {"scalar": 0, "aniso": 1, "r4": 2, "r10": 3}


def clip_job(job):
    """Coverage blocks (scalar vs aniso whitening at the Ronaldo noise
    point) and finding-6 recheck blocks (w_perp bias at ratio 4 vs 10,
    both under aniso whitening)."""
    variant, seed = job
    vi = VARIANT_INDEX[variant]
    rng = np.random.default_rng(800000 + vi * 1000 + seed)
    sc, sa = RATIO_RECHECK.get(variant, CLIP_POINT)
    box = (GOAL_X + rng.normal(0, CROSS_MEAS_SIGMA),
           GOAL_Z + rng.normal(0, CROSS_MEAS_SIGMA), BOX_HALF)
    uv = anisotropic_noise(TRACK["uv_true"], sc, sa, rng)
    noise_px = np.sqrt((sc ** 2 + sa ** 2) / 2) if variant == "scalar" else None
    theta, p0, res = fit_flight(TRACK["times"], uv, CAMERA, box=box,
                                noise_px=noise_px)
    if res.status <= 0:
        return None
    if variant in RATIO_RECHECK:
        return variant, flight_quantities(theta)[3]
    intervals, _ = bootstrap_flight(TRACK["times"], uv, CAMERA, theta, box=box,
                                    noise_px=noise_px, n_boot=N_INNER,
                                    seed=810000 + vi * 1000 + seed)
    hit = (TRUTH_Q >= intervals[:, 0]) & (TRUTH_Q <= intervals[:, 1])
    return variant, hit


def clipmatch():
    jobs = [(v, s) for v in VARIANT_INDEX for s in range(N_CAL)]
    print(f"clip-matched validation at {CLIP_POINT} px (cross, along): "
          f"{len(jobs)} realizations...", flush=True)
    results = {v: [] for v in VARIANT_INDEX}
    with Pool(min(10, os.cpu_count() or 4)) as pool:
        for n, r in enumerate(pool.imap_unordered(clip_job, jobs), 1):
            if r is not None:
                results[r[0]].append(r[1])
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} done", flush=True)

    print(f"\ncoverage at {CLIP_POINT} px (nominal 68%, SE ~5pp at n=80):")
    print(f"{'quantity':<18}{'scalar whiten':>15}{'aniso whiten':>15}")
    cov_s = np.mean(results["scalar"], axis=0)
    cov_a = np.mean(results["aniso"], axis=0)
    for i, name in enumerate(QUANTITY_NAMES):
        print(f"{name:<18}{100*cov_s[i]:>14.0f}%{100*cov_a[i]:>14.0f}%")
    print(f"  ({len(results['scalar'])}/{N_CAL} scalar, "
          f"{len(results['aniso'])}/{N_CAL} aniso converged)")

    print(f"\nfinding-6 recheck under aniso whitening "
          f"(truth w_perp {TRUTH_Q[3]:.0f} rpm):")
    for v, (sc, sa) in RATIO_RECHECK.items():
        est = np.array(results[v])
        print(f"  cross {sc} px ratio {sa/sc:.0f}: median w_perp "
              f"{np.median(est):.0f} rpm, bias {np.median(est)-TRUTH_Q[3]:+.0f} "
              f"({len(est)}/{N_CAL})")


if __name__ == "__main__":
    {"calibrate": calibrate, "verify": verify,
     "clipmatch": clipmatch}[sys.argv[1]]()
