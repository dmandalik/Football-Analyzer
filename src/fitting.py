"""Fit flight parameters to an observed 2D pixel track.

Model, as validated by the Phase 0 synthetic experiments:
  - the ball starts on the turf, so z0 is fixed at ball radius and x0/y0
    follow from intersecting the first observed pixel's camera ray with
    that plane — launch position is derived, not fitted;
  - free parameters are launch velocity (3) and spin (3, rad/s);
  - the goal-plane crossing enters as a measurement in the residual
    (hinge box), not a post-hoc discard filter;
  - uncertainty comes from a parametric percentile bootstrap, with spin
    intervals inflated by a synthetic-calibrated factor.
"""

import numpy as np
from scipy.optimize import least_squares

from src.physics import RADIUS, simulate

RAD_TO_RPM = 60.0 / (2.0 * np.pi)

# free parameters theta6 = [vx, vy, vz, wx, wy, wz]
LOWER6 = np.array([-50.0, -50.0, -20.0, -250.0, -250.0, -250.0])
UPPER6 = np.array([50.0, 10.0, 30.0, 250.0, 250.0, 250.0])
DEFAULT_GUESS6 = np.array([0.0, -23.0, 6.0, 0.0, 0.0, 0.0])

ENDPOINT_HINGE_WEIGHT = 0.05  # m, hinge stiffness outside the crossing box

# Spin correction, keyed by each fit's own measured reprojection RMS.
# The estimator shrinks w_perp toward zero as tracking noise grows, and a
# bootstrap recentres on the biased estimate, so intervals fail by BIAS,
# not width: after recentring by b(rms), near-nominal coverage needs almost
# no inflation. Tables from experiment_spin_calibration (80 realizations
# per level, 25 fps, 0.3 m crossing box measured to 0.2 m); linear
# interpolation between levels, clamped at the ends. Re-calibrate if frame
# rate, camera geometry, or the constraint config changes materially.
SPIN_CAL_RMS = np.array([0.49, 0.96, 1.87, 2.86, 4.85])
SPIN_CAL_BIAS = np.array([-1.3, -24.8, -57.6, -194.8, -290.0])   # rpm
SPIN_CAL_INFLATION = np.array([1.10, 1.00, 1.00, 1.00, 1.15])

# w_par is unobservable; its interval keeps the legacy constant inflation.
SPIN_INTERVAL_INFLATION = 1.5

QUANTITY_NAMES = ["speed [m/s]", "elevation [deg]", "azimuth [deg]",
                  "w_perp [rpm]", "w_par [rpm]"]
SPIN_ROWS = [3, 4]


def spin_correction(rms):
    """(bias_rpm, inflation) for the w_perp interval at a measured
    reprojection RMS. Corrected quantity: estimate - bias."""
    return (float(np.interp(rms, SPIN_CAL_RMS, SPIN_CAL_BIAS)),
            float(np.interp(rms, SPIN_CAL_RMS, SPIN_CAL_INFLATION)))


def launch_point(uv0, camera, z0=RADIUS):
    """Launch position: the first observed pixel's camera ray intersected
    with the plane z = z0 (ball resting on the grass)."""
    ray = camera.R.T @ np.linalg.solve(camera.K, np.array([uv0[0], uv0[1], 1.0]))
    t = (z0 - camera.C[2]) / ray[2]
    return camera.C + t * ray


def crossing_hinge(p0, v0, omega, box):
    """Hinge residuals on the goal-plane (y=0) crossing: zero when the
    crossing lies inside the measured box (cx, cz, half_width)."""
    cx, cz, half = box
    xyz = simulate(p0, v0, omega, np.linspace(0.0, 1.6, 100))
    below = xyz[:, 1] <= 0.0
    if not below.any():
        return np.array([20.0, 20.0])  # never reaches the goal plane
    i = np.argmax(below)
    a, b = xyz[i - 1], xyz[i]
    cross = a + (a[1] / (a[1] - b[1])) * (b - a)
    ex = max(0.0, abs(cross[0] - cx) - half)
    ez = max(0.0, abs(cross[2] - cz) - half)
    return np.array([ex, ez]) / ENDPOINT_HINGE_WEIGHT


def fit_flight(times, uv_obs, camera, box=None, noise_px=2.0, guess=None):
    """Fit launch velocity and spin to a pixel track.

    box: optional (cx, cz, half_width) goal-plane crossing measurement.
    Set half_width no tighter than the crossing measurement's real accuracy
    — a mis-centred tight box produces confident wrong answers.
    Returns (theta6, p0, result).
    """
    p0 = launch_point(uv_obs[0], camera)

    def f(theta6):
        v0, omega = theta6[:3], theta6[3:]
        uv = camera.project(simulate(p0, v0, omega, times))
        r = ((uv - uv_obs) / noise_px).ravel()
        if box is not None:
            r = np.concatenate([r, crossing_hinge(p0, v0, omega, box)])
        return r

    result = least_squares(f, DEFAULT_GUESS6 if guess is None else guess,
                           bounds=(LOWER6, UPPER6), method="trf", x_scale="jac")
    return result.x, p0, result


def reprojection_rms(theta6, p0, times, uv_obs, camera):
    """Measured tracking-noise proxy: component-wise rms pixel residual of
    the fitted flight against the observed track. Keys the spin correction."""
    uv = camera.project(simulate(p0, theta6[:3], theta6[3:], times))
    return float(np.sqrt(np.mean((uv - uv_obs) ** 2)))


def flight_quantities(theta6):
    """speed, elevation, azimuth (vs straight-at-goal -y), and spin
    decomposed about the instantaneous launch velocity direction v0/|v0|:
    w_par = w . v_hat (signed; produces no Magnus force, structurally
    unobservable from the trajectory), w_perp = |w - w_par*v_hat|
    (transverse; produces all the Magnus force)."""
    v0, omega = theta6[:3], theta6[3:]
    speed = np.linalg.norm(v0)
    v_hat = v0 / speed
    w_par = omega @ v_hat
    w_perp = np.linalg.norm(omega - w_par * v_hat)
    return np.array([
        speed,
        np.degrees(np.arctan2(v0[2], np.hypot(v0[0], v0[1]))),
        np.degrees(np.arctan2(v0[0], -v0[1])),
        w_perp * RAD_TO_RPM,
        w_par * RAD_TO_RPM,
    ])


def bootstrap_flight(times, uv_obs, camera, theta6, box=None, noise_px=2.0,
                     n_boot=19, seed=0):
    """Parametric percentile bootstrap around a fitted flight.

    Synthetic tracks are generated from the fitted trajectory plus fresh
    pixel noise and refit with the same procedure (same box). Returns
    (intervals, samples): intervals is (5, 2) of [lo, hi] per quantity in
    QUANTITY_NAMES. The w_perp row is recentred by -b(rms) and inflated by
    k(rms) from the fit's own measured reprojection RMS (spin_correction);
    the unobservable w_par row keeps the constant inflation. samples is a
    list of (theta6, p0) bootstrap fits for rendering fans.
    """
    p0 = launch_point(uv_obs[0], camera)
    uv_clean = camera.project(simulate(p0, theta6[:3], theta6[3:], times))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        uv = uv_clean + rng.normal(0.0, noise_px, size=uv_clean.shape)
        th, p0_b, res = fit_flight(times, uv, camera, box=box, noise_px=noise_px)
        if res.status > 0:
            samples.append((th, p0_b))

    q = np.array([flight_quantities(th) for th, _ in samples])
    lo, hi = np.percentile(q, [16, 84], axis=0)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    bias, inflation = spin_correction(reprojection_rms(theta6, p0, times,
                                                       uv_obs, camera))
    centre[3] -= bias
    half[3] *= inflation
    half[4] *= SPIN_INTERVAL_INFLATION
    return np.stack([centre - half, centre + half], axis=1), samples


# --- legacy 9-parameter interface, kept for the Phase 0 experiment scripts

LOWER = np.array([-50.0, 2.0, 0.0, -50.0, -50.0, -20.0, -250.0, -250.0, -250.0])
UPPER = np.array([50.0, 45.0, 1.0, 50.0, 10.0, 30.0, 250.0, 250.0, 250.0])


def unpack(theta):
    theta = np.asarray(theta, dtype=float)
    return theta[0:3], theta[3:6], theta[6:9]


def residuals(theta, times, uv_obs, camera):
    p0, v0, omega = unpack(theta)
    xyz = simulate(p0, v0, omega, times)
    return (camera.project(xyz) - uv_obs).ravel()


def default_guess():
    return np.array([0.0, 20.0, 0.11, 0.0, -23.0, 6.0, 0.0, 0.0, 0.0])
