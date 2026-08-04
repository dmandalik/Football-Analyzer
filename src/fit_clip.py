"""End-to-end flight fit for one real clip: track + calibration -> launch
parameters with calibrated intervals, goal-mouth check, saved fit JSON.

The goal-plane crossing is a MEASURED quantity: pass the ball's pixel at
the frame where it crosses the goal plane; it is back-projected onto the
y=0 plane to centre the hinge box. Choose --half conservatively (wide
enough to certainly contain truth): offset boxes were the failure mode in
the stress test, oversized ones only cost a little precision.

Run from repo root, e.g.:
  python -m src.fit_clip ronaldo_spain_2018 \
      --track data/tracks/ronaldo_kick.csv \
      --calib data/calibrations/ronaldo_frame0.json \
      --window 30 90 --fps 50 --crossing 1445 470 90 --half 0.5
"""

import argparse
import json
import os

import numpy as np

from src.synthetic import Camera, GOAL_HALF_WIDTH, GOAL_HEIGHT
from src.tracking import load_track
from src.fitting import (QUANTITY_NAMES, bootstrap_flight, fit_flight,
                         flight_quantities, residual_decomposition,
                         spin_correction, simulate)


def load_camera(calib_path):
    d = json.load(open(calib_path))
    return Camera(np.array(d["K"]), np.array(d["R"]), np.array(d["C"]))


def ground_point(camera, u, v, z=0.0):
    """Back-project a pixel onto the horizontal plane at height z."""
    ray = camera.R.T @ np.linalg.solve(camera.K, np.array([u, v, 1.0]))
    t = (z - camera.C[2]) / ray[2]
    return camera.C + t * ray


def goal_plane_point(camera, u, v):
    """Back-project a pixel onto the goal plane y=0 -> (x, z)."""
    ray = camera.R.T @ np.linalg.solve(camera.K, np.array([u, v, 1.0]))
    t = -camera.C[1] / ray[1]
    p = camera.C + t * ray
    return float(p[0]), float(p[2])


def fitted_crossing(theta6, p0):
    xyz = simulate(p0, theta6[:3], theta6[3:], np.linspace(0.0, 2.5, 400))
    below = xyz[:, 1] <= 0.0
    if not below.any():
        return None
    i = np.argmax(below)
    a, b = xyz[i - 1], xyz[i]
    c = a + (a[1] / (a[1] - b[1])) * (b - a)
    return float(c[0]), float(c[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id")
    ap.add_argument("--track", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--window", type=int, nargs=2, required=True,
                    help="first and last flight frame in the track")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--crossing", type=float, nargs=3, required=True,
                    metavar=("U", "V", "FRAME"),
                    help="ball pixel and frame where it crosses the goal plane")
    ap.add_argument("--half", type=float, default=0.5,
                    help="crossing box half-width [m], conservative")
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--slomo", action="store_true",
                    help="fit the slow-motion factor as a 7th parameter")
    ap.add_argument("--noise", type=float, nargs=2, default=None,
                    metavar=("CROSS", "ALONG"),
                    help="override whitening sigmas, e.g. to fold in "
                         "stabilization error")
    args = ap.parse_args()

    camera = load_camera(args.calib)
    a, b = args.window
    rows = [r for r in load_track(args.track) if r[4] and a <= r[0] <= b]
    times = np.array([(r[0] - a) / args.fps for r in rows])
    uv = np.array([[r[1], r[2]] for r in rows])
    print(f"{len(rows)} tracked flight frames over {times[-1]:.2f} s")

    cx, cz = goal_plane_point(camera, args.crossing[0], args.crossing[1])
    box = (cx, cz, args.half)
    in_goal = abs(cx) < GOAL_HALF_WIDTH and 0 < cz < GOAL_HEIGHT
    print(f"measured crossing: x={cx:+.2f}, z={cz:.2f} m +/- {args.half} "
          f"({'inside' if in_goal else 'OUTSIDE'} the goal mouth)")

    theta, p0, res = fit_flight(times, uv, camera, box=box, noise=args.noise,
                                slomo=args.slomo)
    t_true = times / theta[6] if args.slomo else times
    rms_cross, rms_along = residual_decomposition(theta[:6], p0, t_true, uv,
                                                  camera)
    print(f"fit: status {res.status}, nfev {res.nfev}; residuals "
          f"cross {rms_cross:.2f} px, along {rms_along:.2f} px "
          f"(ratio {rms_along / rms_cross:.1f})")
    print(f"launch point (first-frame ray): ({p0[0]:+.2f}, {p0[1]:.2f}, "
          f"{p0[2]:.2f}) m -> {np.hypot(p0[0], p0[1]):.1f} m out")

    intervals, samples = bootstrap_flight(times, uv, camera, theta, box=box,
                                          n_boot=args.n_boot, seed=0,
                                          slomo=args.slomo)
    bias, _ = spin_correction(rms_cross)
    q = flight_quantities(theta[:6])
    q[3] -= bias
    print(f"\n{'quantity':<18}{'estimate':>10}{'68% interval':>20}")
    for i, name in enumerate(QUANTITY_NAMES):
        note = ("  (debiased)" if i == 3 else
                "  (UNOBSERVABLE)" if i == 4 else "")
        print(f"{name:<18}{q[i]:>10.2f}"
              f"{f'[{intervals[i, 0]:.2f}, {intervals[i, 1]:.2f}]':>20}{note}")
    if args.slomo:
        print(f"{'slow-mo factor s':<18}{theta[6]:>10.2f}"
              f"{f'[{intervals[5, 0]:.2f}, {intervals[5, 1]:.2f}]':>20}"
              f"  (broadcast multiples: 2, 2.5, 3)")

    fc = fitted_crossing(theta[:6], p0)
    if fc:
        miss = np.hypot(fc[0] - cx, fc[1] - cz)
        print(f"\ngoal-mouth check: fitted crossing x={fc[0]:+.2f}, z={fc[1]:.2f} "
              f"vs measured x={cx:+.2f}, z={cz:.2f} -> {miss:.2f} m "
              f"({'inside box' if miss < args.half else 'AT/OUTSIDE box edge'})")

    os.makedirs("data/fits", exist_ok=True)
    out = f"data/fits/{args.clip_id}.json"
    json.dump({
        "clip_id": args.clip_id, "theta6": theta.tolist(), "p0": p0.tolist(),
        "quantities": dict(zip(QUANTITY_NAMES, q.tolist())),
        "intervals": intervals.tolist(), "box": list(box),
        "rms_cross": rms_cross, "rms_along": rms_along,
        "window": [a, b], "fps": args.fps,
        "samples": [[t.tolist(), p.tolist()] for t, p in samples],
        "calib": args.calib, "track": args.track,
    }, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
