"""Stress test of the endpoint constraint (config D of the constraint
experiment), which was optimistic: its crossing box was centred on truth.

Part 1 — sweep, 50 noise realizations each at 2 px:
  - box half-width 0.1 / 0.3 / 0.5 / 1.0 m, centred on the true crossing
  - box half-width 0.3 m, centre offset from truth by 0.2 / 0.5 m,
    horizontally (dx) and vertically (dz), separately
  - "goal frame only": crossing anywhere inside the 7.32 x 2.44 m frame
  For each: 68% interval widths and bias (median - truth) per metric.
  Key question: does an offset box give tight intervals around a wrong answer?

Part 2 — coverage, 200 realizations: half-width 0.3 m box whose centre
carries a fresh N(0, 0.2 m) measurement error per realization (the realistic
case: on real footage the crossing is itself measured). Each realization
gets the interval the pipeline would actually report — a parametric
bootstrap: refit 19 synthetic tracks generated from the fitted trajectory —
and we count how often that interval contains truth, per parameter. Also
counts +/-1 sigma linearized-covariance coverage for comparison.

Base model everywhere (as in config D): z0 pinned at 0.11 m, Gaussian prior
sigma = 0.2 m on x0, y0 centred on truth.

Run from repo root:  python -m src.experiment_endpoint_stress
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from src.physics import simulate
from src.synthetic import GOAL_HALF_WIDTH, GOAL_HEIGHT
from src.fitting import residuals, default_guess, unpack
from src.validate_recovery import P0_TRUE, V0_TRUE, OMEGA_TRUE, NOISE_PX
from src.experiment_constraints import (CAMERA, TRACK, GOAL_X, GOAL_Z, IDX8,
                                        LOWER8, UPPER8, PRIOR_SIGMA_XY,
                                        to_theta9, metrics, METRIC_NAMES)

N_SWEEP = 50
N_COVERAGE = 200
N_INNER = 19              # bootstrap refits per coverage realization
CROSS_MEAS_SIGMA = 0.2    # m, crossing measurement error in part 2
HINGE_WEIGHT = 0.05       # m, hinge stiffness outside the box

TRUTH9 = np.concatenate([P0_TRUE, V0_TRUE, OMEGA_TRUE])
PARAM8_NAMES = ["x0", "y0", "vx0", "vy0", "vz0", "wx", "wy", "wz"]

# box = (cx, cz, half_width); None means "goal frame only"
SWEEP = [
    ("w=0.10", (GOAL_X, GOAL_Z, 0.10)),
    ("w=0.30 (=D)", (GOAL_X, GOAL_Z, 0.30)),
    ("w=0.50", (GOAL_X, GOAL_Z, 0.50)),
    ("w=1.00", (GOAL_X, GOAL_Z, 1.00)),
    ("w=0.30 dx=0.2", (GOAL_X + 0.2, GOAL_Z, 0.30)),
    ("w=0.30 dx=0.5", (GOAL_X + 0.5, GOAL_Z, 0.30)),
    ("w=0.30 dz=0.2", (GOAL_X, GOAL_Z + 0.2, 0.30)),
    ("w=0.30 dz=0.5", (GOAL_X, GOAL_Z + 0.5, 0.30)),
    ("goal frame only", None),
]


def crossing_of(theta9):
    p0, v0, omega = unpack(theta9)
    xyz = simulate(p0, v0, omega, np.linspace(0.0, 1.6, 100))
    below = xyz[:, 1] <= 0.0
    if not below.any():
        return None
    i = np.argmax(below)
    a, b = xyz[i - 1], xyz[i]
    return a + (a[1] / (a[1] - b[1])) * (b - a)


def endpoint_hinge(theta9, box):
    cross = crossing_of(theta9)
    if cross is None:
        return np.array([20.0, 20.0])
    if box is None:  # inside the goal frame, no position measurement
        ex = max(0.0, abs(cross[0]) - GOAL_HALF_WIDTH)
        ez = max(0.0, cross[2] - GOAL_HEIGHT, -cross[2])
    else:
        cx, cz, half = box
        ex = max(0.0, abs(cross[0] - cx) - half)
        ez = max(0.0, abs(cross[2] - cz) - half)
    return np.array([ex, ez]) / HINGE_WEIGHT


def fit_track(uv_obs, box):
    """Config-D-style fit: 8 free params, whitened pixels + launch prior +
    endpoint hinge. Returns (theta9, status, sigma8) with sigma8 from the
    linearized covariance (whitened units, so cov = (J^T J)^-1)."""

    def f(theta8):
        theta9 = to_theta9(theta8)
        return np.concatenate([
            residuals(theta9, TRACK["times"], uv_obs, CAMERA) / NOISE_PX,
            (theta9[:2] - P0_TRUE[:2]) / PRIOR_SIGMA_XY,
            endpoint_hinge(theta9, box),
        ])

    res = least_squares(f, default_guess()[IDX8], bounds=(LOWER8, UPPER8),
                        method="trf", x_scale="jac")
    cov8 = np.linalg.pinv(res.jac.T @ res.jac)
    sigma8 = np.sqrt(np.maximum(np.diag(cov8), 0.0))
    return to_theta9(res.x), int(res.status), sigma8


def noisy_track(uv_clean, seed):
    rng = np.random.default_rng(seed)
    return uv_clean + rng.normal(0.0, NOISE_PX, size=uv_clean.shape)


def run_sweep_job(job):
    idx, seed = job
    _, box = SWEEP[idx]
    theta9, status, _ = fit_track(noisy_track(TRACK["uv_true"], 3000 + seed), box)
    return idx, theta9, status


def run_outer_job(seed):
    """One coverage realization: noisy box centre + noisy track + fit."""
    rng = np.random.default_rng(7000 + seed)
    cx, cz = GOAL_X + rng.normal(0, CROSS_MEAS_SIGMA), GOAL_Z + rng.normal(0, CROSS_MEAS_SIGMA)
    uv = TRACK["uv_true"] + rng.normal(0.0, NOISE_PX, size=TRACK["uv_true"].shape)
    theta9, status, sigma8 = fit_track(uv, (cx, cz, 0.30))
    return seed, theta9, status, sigma8, (cx, cz)


def run_inner_job(job):
    """One parametric-bootstrap refit: synthetic track from the outer fit's
    trajectory, same fitting procedure. With jitter=True the box centre is
    resampled from N(measured, CROSS_MEAS_SIGMA) per refit, so the crossing
    measurement error propagates into the bootstrap intervals instead of
    acting as an invisible shared systematic."""
    outer_seed, inner_seed, uv_clean, cx, cz, jitter = job
    uv = noisy_track(uv_clean, 9000 + outer_seed * 100 + inner_seed)
    if jitter:
        rng = np.random.default_rng(40000 + outer_seed * 100 + inner_seed)
        cx += rng.normal(0, CROSS_MEAS_SIGMA)
        cz += rng.normal(0, CROSS_MEAS_SIGMA)
    theta9, status, _ = fit_track(uv, (cx, cz, 0.30))
    return outer_seed, theta9, status


def print_sweep(results):
    m_true = metrics(TRUTH9)
    per_config = {i: [] for i in range(len(SWEEP))}
    for idx, theta9, status in results:
        if status > 0:
            per_config[idx].append(metrics(theta9))
    for title, values in [("68% interval WIDTHS", "width"), ("BIAS (median - truth)", "bias")]:
        print(f"\n{title}:")
        print(f"{'config':<18}" + "".join(f"{n:>16}" for n in METRIC_NAMES))
        for i, (name, _) in enumerate(SWEEP):
            m = np.array(per_config[i])
            lo, hi = np.percentile(m, [16, 84], axis=0)
            vals = (hi - lo) if values == "width" else (np.median(m, axis=0) - m_true)
            print(f"{name:<18}" + "".join(f"{v:>16.2f}" for v in vals)
                  + f"   ({len(m)}/{N_SWEEP})")


def print_coverage(outer, inner):
    boots = {seed: [] for seed, *_ in outer}
    for outer_seed, theta9, status in inner:
        if status > 0:
            boots[outer_seed].append(theta9)

    truth8 = TRUTH9[IDX8]
    m_true = metrics(TRUTH9)[:5]
    hits_boot8, hits_sigma8, hits_boot_m = [], [], []
    used = 0
    for seed, theta9, status, sigma8, _ in outer:
        b = boots[seed]
        if status <= 0 or len(b) < 10:
            continue
        used += 1
        b = np.array(b)
        lo8, hi8 = np.percentile(b[:, IDX8], [16, 84], axis=0)
        hits_boot8.append((truth8 >= lo8) & (truth8 <= hi8))
        hits_sigma8.append(np.abs(theta9[IDX8] - truth8) <= sigma8)
        bm = np.array([metrics(t)[:5] for t in b])
        lom, him = np.percentile(bm, [16, 84], axis=0)
        hits_boot_m.append((m_true >= lom) & (m_true <= him))

    print(f"\nCOVERAGE ({used}/{N_COVERAGE} realizations usable; nominal 68%):")
    print(f"{'parameter':<12}{'bootstrap 68%':>16}{'+/-1sigma cov':>16}")
    boot8, sig8 = np.mean(hits_boot8, axis=0), np.mean(hits_sigma8, axis=0)
    for i, name in enumerate(PARAM8_NAMES):
        print(f"{name:<12}{100*boot8[i]:>15.0f}%{100*sig8[i]:>15.0f}%")
    bootm = np.mean(hits_boot_m, axis=0)
    for i, name in enumerate(METRIC_NAMES[:5]):
        print(f"{name:<12}{100*bootm[i]:>15.0f}%{'-':>16}")


def analyze(path):
    """Percentile vs basic (bias-flipping) bootstrap intervals, from saved
    fits — no refitting. Basic: [2*theta_hat - q84, 2*theta_hat - q16]."""
    d = np.load(path)
    truth8, m_true = TRUTH9[IDX8], metrics(TRUTH9)[:5]
    hits = {k: [] for k in ["pct8", "bas8", "pctm", "basm"]}
    used = 0
    for i in range(len(d["outer_theta"])):
        sel = (d["inner_seed"] == i) & (d["inner_status"] > 0)
        if d["outer_status"][i] <= 0 or sel.sum() < 10:
            continue
        used += 1
        th8 = d["outer_theta"][i][IDX8]
        b8 = d["inner_theta"][sel][:, IDX8]
        lo, hi = np.percentile(b8, [16, 84], axis=0)
        hits["pct8"].append((truth8 >= lo) & (truth8 <= hi))
        hits["bas8"].append((truth8 >= 2 * th8 - hi) & (truth8 <= 2 * th8 - lo))
        thm = metrics(d["outer_theta"][i])[:5]
        bm = np.array([metrics(t)[:5] for t in d["inner_theta"][sel]])
        lom, him = np.percentile(bm, [16, 84], axis=0)
        hits["pctm"].append((m_true >= lom) & (m_true <= him))
        hits["basm"].append((m_true >= 2 * thm - him) & (m_true <= 2 * thm - lom))
    print(f"COVERAGE from {path} ({used} realizations; nominal 68%):")
    print(f"{'parameter':<14}{'percentile':>12}{'basic':>12}")
    for names, pk, bk in [(PARAM8_NAMES, "pct8", "bas8"),
                          (METRIC_NAMES[:5], "pctm", "basm")]:
        p, b = np.mean(hits[pk], axis=0), np.mean(hits[bk], axis=0)
        for j, name in enumerate(names):
            print(f"{name:<14}{100*p[j]:>11.0f}%{100*b[j]:>11.0f}%")


def main():
    # modes: "all" = sweep + plain coverage;  "coverage" = plain coverage
    # only;  "jitter" = coverage only with the box centre resampled per
    # bootstrap refit (same outer seeds throughout, so comparisons isolate
    # one change at a time);  "analyze <npz>" = interval flavours from
    # saved fits.
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "analyze":
        return analyze(sys.argv[2])
    jitter = mode == "jitter"
    print(f"true crossing: x={GOAL_X:+.2f}, z={GOAL_Z:.2f}; "
          f"truth metrics: " + "  ".join(
              f"{n}={v:.1f}" for n, v in zip(METRIC_NAMES, metrics(TRUTH9))))

    pool = Pool(min(10, os.cpu_count() or 4))
    if not jitter:
        sweep_jobs = [(i, s) for i in range(len(SWEEP)) for s in range(N_SWEEP)]
        print(f"\npart 1: sweep, {len(sweep_jobs)} fits...")
        sweep_results = pool.map(run_sweep_job, sweep_jobs)
        print_sweep(sweep_results)

    print(f"\npart 2: coverage ({'jittered' if jitter else 'plain'} bootstrap), "
          f"{N_COVERAGE} outer + {N_COVERAGE * N_INNER} bootstrap fits...")
    outer = pool.map(run_outer_job, range(N_COVERAGE))
    inner_jobs = []
    for seed, theta9, status, _, (cx, cz) in outer:
        if status > 0:
            uv_clean = CAMERA.project(simulate(*unpack(theta9), TRACK["times"]))
            inner_jobs.extend((seed, k, uv_clean, cx, cz, jitter)
                              for k in range(N_INNER))
    inner = pool.map(run_inner_job, inner_jobs)
    pool.close()

    os.makedirs("reports", exist_ok=True)
    out = f"reports/coverage_fits_{mode}.npz"
    np.savez(out,
             outer_theta=np.array([t for _, t, *_ in outer]),
             outer_status=np.array([s for _, _, s, *_ in outer]),
             centers=np.array([c for *_, c in outer]),
             inner_seed=np.array([s for s, _, _ in inner]),
             inner_theta=np.array([t for _, t, _ in inner]),
             inner_status=np.array([s for _, _, s in inner]))
    print(f"fits saved to {out}")
    print_coverage(outer, inner)


if __name__ == "__main__":
    main()
