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
from scipy.signal import savgol_filter

from src.physics import RADIUS, simulate

RAD_TO_RPM = 60.0 / (2.0 * np.pi)

# free parameters theta6 = [vx, vy, vz, wx, wy, wz]
LOWER6 = np.array([-50.0, -50.0, -20.0, -250.0, -250.0, -250.0])
UPPER6 = np.array([50.0, 10.0, 30.0, 250.0, 250.0, 250.0])
DEFAULT_GUESS6 = np.array([0.0, -23.0, 6.0, 0.0, 0.0, 0.0])

ENDPOINT_HINGE_WEIGHT = 0.05  # m, hinge stiffness outside the crossing box

# Spin correction, keyed by each fit's measured CROSS-TRACK residual RMS.
# Real tracking noise is anisotropic — blur smears the ball centroid along
# its travel direction 3-11x more than across it — and curvature (spin)
# information lives cross-track, so that is the axis that gates spin. The
# estimator shrinks w_perp toward zero as noise grows and a bootstrap
# recentres on the biased estimate, so intervals fail by BIAS, not width:
# after recentring by b(rms_cross), near-nominal coverage needs little
# inflation. Tables from experiment_spin_calibration (anisotropic noise,
# along/cross ratio 4, 80 realizations per level, 25 fps, 0.3 m crossing
# box measured to 0.2 m); linear interpolation, clamped at the ends.
# Re-calibrate if frame rate, geometry, or constraint config changes.
# NOTE: this table was derived under legacy SCALAR whitening. Under the
# anisotropic-whitening default the ratio dependence disappears and the
# bias shrinks (clipmatch recheck: -86 rpm at ratio 4 vs -58 at ratio 10,
# statistically indistinguishable; scalar had -95 vs -250) — the table is
# approximately right near 1.5-2 px cross-track but should be re-derived
# under whitening. For fit-grade clips, prefer a clip-matched calibration
# at the clip's own measured (cross, along) noise; table is for grading.
SPIN_CAL_RMS = np.array([0.51, 1.01, 1.60, 2.10, 3.00])
SPIN_CAL_BIAS = np.array([6.6, -165.3, -95.3, -151.2, -94.4])    # rpm
SPIN_CAL_INFLATION = np.array([1.25, 1.65, 1.45, 1.05, 1.55])

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


SLOMO_BOUNDS = (1.0, 6.0)
SLOMO_GUESS = 2.5


def fit_flight(times, uv_obs, camera, box=None, noise_px=None, noise=None,
               guess=None, slomo=False):
    """Fit launch velocity and spin to a pixel track.

    Whitening (default): residuals are rotated per-frame into the local
    track frame — axes fixed from the SG-smoothed observed track, see
    estimate_track_noise — and scaled by 1/sigma_cross, 1/sigma_along.
    Pass noise=(sigma_cross, sigma_along) to override the estimated
    sigmas, or noise_px=<scalar> for legacy isotropic whitening.

    box: optional (cx, cz, half_width) goal-plane crossing measurement.
    Set half_width no tighter than the crossing measurement's real accuracy
    — a mis-centred tight box produces confident wrong answers.

    slomo=True appends the slow-motion factor s as a 7th parameter: the
    observed timestamps are s times slower than true time, so the physics
    runs on times/s. Identifiable because apparent gravity (g/s^2) is
    measured directly by image curvature over the arc.
    Returns (theta, p0, result) — theta has 7 entries when slomo.
    """
    p0 = launch_point(uv_obs[0], camera)
    if noise_px is None:
        sc, sa, t_hat, n_hat = estimate_track_noise(uv_obs)
        if noise is not None:
            sc, sa = max(SIGMA_FLOOR, noise[0]), max(SIGMA_FLOOR, noise[1])

    def f(theta):
        v0, omega = theta[:3], theta[3:6]
        t_true = times / theta[6] if slomo else times
        uv = camera.project(simulate(p0, v0, omega, t_true))
        d = uv - uv_obs
        if noise_px is None:
            r = np.concatenate([np.sum(d * n_hat, axis=1) / sc,
                                np.sum(d * t_hat, axis=1) / sa])
        else:
            r = (d / noise_px).ravel()
        if box is not None:
            r = np.concatenate([r, crossing_hinge(p0, v0, omega, box)])
        return r

    if slomo:
        lower = np.append(LOWER6, SLOMO_BOUNDS[0])
        upper = np.append(UPPER6, SLOMO_BOUNDS[1])
        g0 = np.append(DEFAULT_GUESS6, SLOMO_GUESS) if guess is None else guess
    else:
        lower, upper = LOWER6, UPPER6
        g0 = DEFAULT_GUESS6 if guess is None else guess
    result = least_squares(f, g0, bounds=(lower, upper), method="trf",
                           x_scale="jac")
    return result.x, p0, result


def reprojection_rms(theta6, p0, times, uv_obs, camera):
    """Component-wise rms pixel residual of the fitted flight against the
    observed track (isotropic total; see residual_decomposition)."""
    uv = camera.project(simulate(p0, theta6[:3], theta6[3:], times))
    return float(np.sqrt(np.mean((uv - uv_obs) ** 2)))


def track_frame_axes(uv):
    """Per-frame unit tangent and normal of a pixel track."""
    vel = np.gradient(np.asarray(uv, dtype=float), axis=0)
    speed = np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9)
    t_hat = vel / speed
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    return t_hat, n_hat


def residual_decomposition(theta6, p0, times, uv_obs, camera):
    """(rms_cross, rms_along): pixel residuals split normal/tangent to the
    modeled track. Real tracking noise is anisotropic (blur smears the ball
    along its travel direction), and curvature information lives cross-track,
    so rms_cross is what keys the spin correction and clip grading."""
    uv = camera.project(simulate(p0, theta6[:3], theta6[3:], times))
    t_hat, n_hat = track_frame_axes(uv)
    r = uv_obs - uv
    return (float(np.sqrt(np.mean(np.sum(r * n_hat, axis=1) ** 2))),
            float(np.sqrt(np.mean(np.sum(r * t_hat, axis=1) ** 2))))


def anisotropic_noise(uv_clean, sigma_cross, sigma_along, rng):
    """Gaussian pixel noise oriented along the track: sigma_along in the
    travel direction, sigma_cross perpendicular to it."""
    t_hat, n_hat = track_frame_axes(uv_clean)
    n = len(uv_clean)
    return (uv_clean
            + t_hat * rng.normal(0.0, sigma_along, size=(n, 1))
            + n_hat * rng.normal(0.0, sigma_cross, size=(n, 1)))


SG_WINDOW = 9
SIGMA_FLOOR = 0.2  # px; keeps weights finite on near-noiseless tracks


def estimate_track_noise(uv_obs):
    """(sigma_cross, sigma_along, t_hat, n_hat) from the observed track.

    Sigmas come from Savitzky-Golay residuals (window 9, quadratic),
    scaled by sqrt(w/(w-3)) for the dof the smooth absorbs. The per-frame
    axes t_hat/n_hat come from the SG smooth of the OBSERVED track and are
    HELD FIXED during fitting. Do not refactor them to come from the
    projected model trajectory: that makes the whitening depend on the
    parameters being fitted, so a wrong fit can rotate the noise ellipse
    to justify its own errors.
    """
    uv = np.asarray(uv_obs, dtype=float)
    win = SG_WINDOW if len(uv) >= SG_WINDOW else max(5, (len(uv) // 2) * 2 - 1)
    smooth = savgol_filter(uv, window_length=win, polyorder=2, axis=0)
    t_hat, n_hat = track_frame_axes(smooth)
    r = uv - smooth
    scale = np.sqrt(win / (win - 3.0))
    sigma_cross = max(SIGMA_FLOOR,
                      float(np.sqrt(np.mean(np.sum(r * n_hat, axis=1) ** 2))) * scale)
    sigma_along = max(SIGMA_FLOOR,
                      float(np.sqrt(np.mean(np.sum(r * t_hat, axis=1) ** 2))) * scale)
    return sigma_cross, sigma_along, t_hat, n_hat


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


def bootstrap_flight(times, uv_obs, camera, theta6, box=None, noise_px=None,
                     n_boot=19, seed=0, slomo=False):
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
    t_true = times / theta6[6] if slomo else times
    uv_clean = camera.project(simulate(p0, theta6[:3], theta6[3:6], t_true))
    rms_cross, rms_along = residual_decomposition(theta6[:6], p0, t_true,
                                                  uv_obs, camera)
    # fit residuals understate the true noise because the fit absorbs part
    # of it; standard dof inflation (p parameters over n frames)
    p_dim = 7 if slomo else 6
    dof_inflation = np.sqrt(len(times) / max(1.0, len(times) - p_dim))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        uv = anisotropic_noise(uv_clean, rms_cross * dof_inflation,
                               rms_along * dof_inflation, rng)
        th, p0_b, res = fit_flight(times, uv, camera, box=box,
                                   noise_px=noise_px, slomo=slomo)
        if res.status > 0:
            samples.append((th, p0_b))

    q = np.array([flight_quantities(th[:6]) for th, _ in samples])
    if slomo:  # 6th row: the slow-motion factor's own interval
        q = np.hstack([q, np.array([[th[6]] for th, _ in samples])])
    lo, hi = np.percentile(q, [16, 84], axis=0)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    bias, inflation = spin_correction(rms_cross)
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
