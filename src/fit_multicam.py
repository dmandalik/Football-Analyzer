"""End-to-end flight fit with PER-FRAME cameras from goal-PnP poses.

The camera pans and zooms, so every observation gets its own projection
(tripod model: shared position, per-frame rotation and focal). Launch
position comes from the first observation's ground ray; the goal-plane
crossing is measured by back-projecting the last tracked ball pixel onto
y=0 through that frame's camera, with a conservative box.

CLI (from repo root):
  python -m src.fit_multicam fit <clip_id> <track.csv> <poses.npz> \
      <first> <last> --fps 25 [--half 0.6] [--n-boot 30]
  python -m src.fit_multicam render <fit.json> <video> [--out mp4]
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.physics import RADIUS, simulate
from src.tracking import load_track
from src.fitting import (LOWER6, UPPER6, DEFAULT_GUESS6, QUANTITY_NAMES,
                         SPIN_INTERVAL_INFLATION, anisotropic_noise,
                         crossing_hinge, estimate_track_noise,
                         flight_quantities, spin_correction)


def load_cameras(poses_path):
    d = np.load(poses_path)
    cams = {}
    for k, f in enumerate(d["frames"]):
        fk = float(d["fs"][k])
        K = np.array([[fk, 0, 960.0], [0, fk, 540.0], [0, 0, 1.0]])
        R, _ = cv2.Rodrigues(d["rvecs"][k])
        cams[int(f)] = (K, R, d["tvecs"][k].reshape(3), bool(d["ok"][k]))
    return cams


def project_frame(cams, f, pts3):
    K, R, t, _ = cams[f]
    q = (K @ (R @ np.atleast_2d(pts3).T + t.reshape(3, 1))).T
    return q[:, :2] / q[:, 2:]


def ground_ray_point(cams, f, uv, z=RADIUS):
    K, R, t, _ = cams[f]
    C = -R.T @ t
    ray = R.T @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    s = (z - C[2]) / ray[2]
    return C + s * ray


def goal_plane_point(cams, f, uv):
    K, R, t, _ = cams[f]
    C = -R.T @ t
    ray = R.T @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    s = -C[1] / ray[1]
    p = C + s * ray
    return float(p[0]), float(p[2])


def fitted_crossing(theta, p0):
    xyz = simulate(p0, theta[:3], theta[3:], np.linspace(0.0, 2.5, 400))
    below = xyz[:, 1] <= 0
    if not below.any():
        return None
    i = np.argmax(below)
    a, b = xyz[i - 1], xyz[i]
    c = a + (a[1] / (a[1] - b[1])) * (b - a)
    return float(c[0]), float(c[2])


def multicam_fit(times, frames_idx, uv, cams, p0, box, noise):
    sc, sa, t_hat, n_hat = noise

    def resid(theta):
        xyz = simulate(p0, theta[:3], theta[3:], times)
        d = np.array([project_frame(cams, f, xyz[k])[0]
                      for k, f in enumerate(frames_idx)]) - uv
        r = np.concatenate([np.sum(d * n_hat, axis=1) / sc,
                            np.sum(d * t_hat, axis=1) / sa])
        return np.concatenate([r, crossing_hinge(p0, theta[:3], theta[3:], box)])

    return least_squares(resid, DEFAULT_GUESS6, bounds=(LOWER6, UPPER6),
                         method="trf", x_scale="jac")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("clip_id"), f.add_argument("track"), f.add_argument("poses")
    f.add_argument("first", type=int), f.add_argument("last", type=int)
    f.add_argument("--fps", type=float, required=True)
    f.add_argument("--half", type=float, default=0.6)
    f.add_argument("--n-boot", type=int, default=30)
    r = sub.add_parser("render")
    r.add_argument("fit_json"), r.add_argument("video")
    r.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "fit":
        cams = load_cameras(args.poses)
        rows = [t for t in load_track(args.track)
                if t[4] and args.first <= t[0] <= args.last
                and t[0] in cams and cams[t[0]][3]]
        frames_idx = [t[0] for t in rows]
        times = np.array([(t[0] - rows[0][0]) / args.fps for t in rows])
        uv = np.array([[t[1], t[2]] for t in rows])
        print(f"{len(rows)} usable flight observations")

        p0 = ground_ray_point(cams, frames_idx[0], uv[0])
        print(f"launch point (ground ray): ({p0[0]:+.2f}, {p0[1]:.2f}, "
              f"{p0[2]:.2f}) -> {np.hypot(p0[0], p0[1]):.1f} m out")

        cx, cz = goal_plane_point(cams, frames_idx[-1], uv[-1])
        box = (cx, cz, args.half)
        print(f"measured crossing (last tracked pixel, f{frames_idx[-1]}): "
              f"x={cx:+.2f}, z={cz:.2f} +/- {args.half} m")

        noise = estimate_track_noise(uv)
        print(f"track noise: cross {noise[0]:.2f} px, along {noise[1]:.2f} px")

        res = multicam_fit(times, frames_idx, uv, cams, p0, box, noise)
        theta = res.x
        xyz_fit = simulate(p0, theta[:3], theta[3:], times)
        d = np.array([project_frame(cams, f, xyz_fit[k])[0]
                      for k, f in enumerate(frames_idx)]) - uv
        rms = float(np.sqrt(np.mean(d ** 2)))
        print(f"fit: status {res.status}, nfev {res.nfev}, "
              f"reprojection rms {rms:.2f} px")

        # parametric bootstrap with matched anisotropic noise
        uv_clean = uv - d
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(args.n_boot):
            uvb = anisotropic_noise(uv_clean, noise[0], noise[1], rng)
            rb = multicam_fit(times, frames_idx, uvb, cams, p0, box, noise)
            if rb.status > 0:
                samples.append(rb.x)
        q = np.array([flight_quantities(s) for s in samples])
        lo, hi = np.percentile(q, [16, 84], axis=0)
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        bias, infl = spin_correction(noise[0])
        centre[3] -= bias
        half[3] *= infl
        half[4] *= SPIN_INTERVAL_INFLATION
        intervals = np.stack([centre - half, centre + half], axis=1)

        qf = flight_quantities(theta)
        qf[3] -= bias
        print(f"\n{'quantity':<18}{'estimate':>10}{'68% interval':>20}")
        for i, name in enumerate(QUANTITY_NAMES):
            note = ("  (debiased)" if i == 3 else
                    "  (UNOBSERVABLE)" if i == 4 else "")
            print(f"{name:<18}{qf[i]:>10.2f}"
                  f"{f'[{intervals[i,0]:.2f}, {intervals[i,1]:.2f}]':>20}{note}")

        fc = fitted_crossing(theta, p0)
        if fc:
            miss = np.hypot(fc[0] - cx, fc[1] - cz)
            print(f"\ngoal-mouth check: fitted x={fc[0]:+.2f}, z={fc[1]:.2f} "
                  f"vs measured x={cx:+.2f}, z={cz:.2f} -> {miss:.2f} m")

        os.makedirs("data/fits", exist_ok=True)
        out = f"data/fits/{args.clip_id}.json"
        json.dump({"clip_id": args.clip_id, "theta6": theta.tolist(),
                   "p0": p0.tolist(), "window": [frames_idx[0], frames_idx[-1]],
                   "fps": args.fps, "box": list(box),
                   "quantities": dict(zip(QUANTITY_NAMES, qf.tolist())),
                   "intervals": intervals.tolist(),
                   "noise": [noise[0], noise[1]], "reproj_rms": rms,
                   "samples": [s.tolist() for s in samples],
                   "poses": args.poses, "track": args.track},
                  open(out, "w"), indent=1)
        print(f"saved {out}")
        return

    fit = json.load(open(args.fit_json))
    cams = load_cameras(fit["poses"])
    theta = np.array(fit["theta6"])
    p0 = np.array(fit["p0"])
    a, b = fit["window"]
    fps = fit["fps"]
    t_end = (b - a) / fps
    tf = np.linspace(0.0, t_end, 140)
    xyz_best = simulate(p0, theta[:3], theta[3:], tf)
    xyz_samp = [simulate(p0, np.array(s)[:3], np.array(s)[3:], tf)
                for s in fit["samples"][:22]]
    cx, cz, half = fit["box"]
    box3 = np.array([[cx - half, 0, cz - half], [cx + half, 0, cz - half],
                     [cx + half, 0, cz + half], [cx - half, 0, cz + half],
                     [cx - half, 0, cz - half]])

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    out_path = args.out or f"reports/overlays/{fit['clip_id']}_fit_overlay.mp4"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), vfps, (w, h))

    def draw_poly(img, f, xyz, color, thick):
        uvs = project_frame(cams, f, xyz)
        pts = [tuple(np.int32(p)) for p in uvs]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in cams:
            for xs in xyz_samp:
                draw_poly(frame, idx, xs, (200, 200, 80), 1)
            draw_poly(frame, idx, xyz_best, (0, 210, 255), 2)
            draw_poly(frame, idx, box3, (0, 0, 255), 2)
            t = (idx - a) / fps
            if 0 <= t <= t_end:
                pos = simulate(p0, theta[:3], theta[3:], np.array([0.0, t]))[-1]
                u, v = project_frame(cams, idx, pos)[0]
                cv2.circle(frame, (int(u), int(v)), 11, (0, 210, 255), 2)
        vw.write(frame)
        idx += 1
    cap.release()
    vw.release()
    print(f"fit overlay -> {out_path}")


if __name__ == "__main__":
    main()
