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


def smooth_poses(poses_path, out_path, deg=None):
    """A gantry camera moves smoothly; frame-to-frame pose jitter is solve
    noise that aliases into fake ball curvature (i.e. fake spin). Fit
    polynomials to rotation and focal over time — degree chosen by how well
    the GOAL CORNERS reproject through the smoothed poses (over-smoothing
    deletes real pan/zoom, which is worse than the jitter it removes)."""
    from src.goal_pnp import GOAL_3D

    d = np.load(poses_path)
    frames = d["frames"].astype(float)
    ok = d["ok"].astype(bool)
    rv = d["rvecs"].reshape(len(frames), 3)
    C = d["C"]

    def build(deg_try):
        out_rv = np.stack([np.polyval(np.polyfit(frames[ok], rv[ok, c],
                                                 deg_try), frames)
                           for c in range(3)], axis=1)
        fs = np.polyval(np.polyfit(frames[ok], d["fs"][ok], deg_try), frames)
        return out_rv, fs

    # corner reprojection rms through smoothed poses, ok frames only
    side = np.sign(d["corners"][0][0][0] - d["corners"][0][2][0])  # unused
    obj = np.array([v for v in GOAL_3D.values()])
    # recover the x-sign the solve used by testing both against frame 0
    def corner_rms(out_rv, fs, obj_signed):
        errs = []
        for k in np.where(ok)[0]:
            R, _ = cv2.Rodrigues(out_rv[k])
            K = np.array([[fs[k], 0, 960], [0, fs[k], 540], [0, 0, 1.0]])
            q = (K @ (R @ (obj_signed - C).T + (-R @ C).reshape(3, 1)
                      + (R @ C).reshape(3, 1) - (R @ C).reshape(3, 1))).T
            q = (K @ (R @ (obj_signed - C).T)).T
            uv = q[:, :2] / q[:, 2:]
            errs.append(np.sqrt(np.mean((uv - d["corners"][k]) ** 2)))
        return float(np.mean(errs))

    best = None
    for s in (1.0, -1.0):
        obj_s = obj.copy()
        obj_s[:, 0] *= s
        r0 = corner_rms(rv, d["fs"], obj_s)
        if best is None or r0 < best[1]:
            best = (obj_s, r0)
    obj_s, raw_rms = best
    print(f"raw poses corner rms {raw_rms:.2f} px")

    chosen = None
    for deg_try in ([deg] if deg else [3, 4, 5, 6, 7]):
        out_rv, fs = build(deg_try)
        rms = corner_rms(out_rv, fs, obj_s)
        print(f"  deg {deg_try}: corner rms {rms:.2f} px")
        if chosen is None and rms < max(3.5, raw_rms + 1.5):
            chosen = (deg_try, out_rv, fs, rms)
    if chosen is None:
        deg_try = deg or 7
        out_rv, fs = build(deg_try)
        chosen = (deg_try, out_rv, fs, corner_rms(out_rv, fs, obj_s))
    deg_used, out_rv, fs, rms = chosen
    print(f"pose smoothing: degree {deg_used}, corner rms {rms:.2f} px "
          f"(raw {raw_rms:.2f})")
    np.savez(out_path, K=d["K"], frames=d["frames"],
             rvecs=out_rv.reshape(-1, 3, 1),
             tvecs=np.stack([
                 (-cv2.Rodrigues(out_rv[k])[0] @ (-cv2.Rodrigues(rv[k])[0].T
                  @ d["tvecs"][k].reshape(3))).reshape(3, 1)
                 for k in range(len(frames))]),
             fs=fs, C=d["C"], ok=d["ok"], corners=d["corners"])
    print(f"saved {out_path}")


def crossing_frame_geometric(cams, rows):
    """First frame whose ball pixel lies inside the projected goal mouth —
    the measured crossing belongs there, not at the last tracked pixel
    (which may already be behind the plane / in the net)."""
    quad3 = np.array([(-3.66, 0, 0), (-3.66, 0, 2.44),
                      (3.66, 0, 2.44), (3.66, 0, 0)])
    for t in rows[len(rows) // 2:]:
        quad = project_frame(cams, t[0], quad3).astype(np.float32)
        if cv2.pointPolygonTest(quad.reshape(-1, 1, 2),
                                (float(t[1]), float(t[2])), False) >= 0:
            return t
    return rows[-1]


def refined_centers(track_path, first, last):
    """Pick the steadiest center definition per track: mask centroid vs
    principal-axis midpoint (streak endpoints are stabler when the mask
    flickers laterally). Judged by measured along-track noise."""
    import csv as _csv
    cand = {"centroid": [], "axis_mid": []}
    frames = []
    with open(track_path) as fh:
        for r in _csv.DictReader(fh):
            f = int(r["frame"])
            if not (first <= f <= last and int(r["ok"])):
                continue
            frames.append(f)
            cand["centroid"].append((float(r["u"]), float(r["v"])))
            if r.get("ax_lo_u") not in (None, "", "nan"):
                mid = ((float(r["ax_lo_u"]) + float(r["ax_hi_u"])) / 2,
                       (float(r["ax_lo_v"]) + float(r["ax_hi_v"])) / 2)
            else:
                mid = cand["centroid"][-1]
            cand["axis_mid"].append(mid)
    best, best_sa = None, None
    for name, uv in cand.items():
        sc, sa, *_ = estimate_track_noise(np.array(uv))
        print(f"  center={name}: cross {sc:.2f} px, along {sa:.2f} px")
        if best_sa is None or sa < best_sa:
            best, best_sa = name, sa
    print(f"  using {best}")
    return frames, np.array(cand[best])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smooth-poses")
    s.add_argument("poses"), s.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("clip_id"), f.add_argument("track"), f.add_argument("poses")
    f.add_argument("first", type=int), f.add_argument("last", type=int)
    f.add_argument("--fps", type=float, required=True)
    f.add_argument("--half", type=float, default=0.6)
    f.add_argument("--n-boot", type=int, default=30)
    f.add_argument("--pose-noise", type=float, default=0.0,
                   help="pose-error floor [px], added in quadrature")
    f.add_argument("--refined", action="store_true",
                   help="choose mask-axis center vs centroid by along-noise")
    f.add_argument("--geom-crossing", action="store_true",
                   help="measure crossing at the goal-mouth entry frame")
    r = sub.add_parser("render")
    r.add_argument("fit_json"), r.add_argument("video")
    r.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "smooth-poses":
        smooth_poses(args.poses, args.out)
        return

    if args.cmd == "fit":
        cams = load_cameras(args.poses)
        if args.refined:
            frames_all, uv_all = refined_centers(args.track, args.first,
                                                 args.last)
            keep = [k for k, f in enumerate(frames_all)
                    if f in cams and cams[f][3]]
            frames_idx = [frames_all[k] for k in keep]
            uv = uv_all[keep]
            rows = [(f, uv[k][0], uv[k][1], 0, 1)
                    for k, f in enumerate(frames_idx)]
        else:
            rows = [t for t in load_track(args.track)
                    if t[4] and args.first <= t[0] <= args.last
                    and t[0] in cams and cams[t[0]][3]]
            frames_idx = [t[0] for t in rows]
            uv = np.array([[t[1], t[2]] for t in rows])
        times = np.array([(f - frames_idx[0]) / args.fps for f in frames_idx])
        print(f"{len(frames_idx)} usable flight observations")

        p0 = ground_ray_point(cams, frames_idx[0], uv[0])
        print(f"launch point (ground ray): ({p0[0]:+.2f}, {p0[1]:.2f}, "
              f"{p0[2]:.2f}) -> {np.hypot(p0[0], p0[1]):.1f} m out")

        if args.geom_crossing:
            ct = crossing_frame_geometric(cams, rows)
            cx, cz = goal_plane_point(cams, ct[0], (ct[1], ct[2]))
            print(f"measured crossing (goal-mouth entry, f{ct[0]}): "
                  f"x={cx:+.2f}, z={cz:.2f} +/- {args.half} m")
        else:
            cx, cz = goal_plane_point(cams, frames_idx[-1], uv[-1])
            print(f"measured crossing (last tracked pixel, f{frames_idx[-1]}): "
                  f"x={cx:+.2f}, z={cz:.2f} +/- {args.half} m")
        box = (cx, cz, args.half)

        sc, sa, t_hat, n_hat = estimate_track_noise(uv)
        sc = float(np.hypot(sc, args.pose_noise))
        sa = float(np.hypot(sa, args.pose_noise))
        noise = (sc, sa, t_hat, n_hat)
        print(f"track noise incl. pose floor {args.pose_noise} px: "
              f"cross {sc:.2f} px, along {sa:.2f} px")

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
