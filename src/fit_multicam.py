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
    cx, cy = float(d["K"][0, 2]), float(d["K"][1, 2])  # clip resolution varies
    cams = {}
    for k, f in enumerate(d["frames"]):
        fk = float(d["fs"][k])
        K = np.array([[fk, 0, cx], [0, fk, cy], [0, 0, 1.0]])
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


def simulate_at(p0, v0, omega, times):
    """simulate() pins (p0, v0) at times[0]; when the kick precedes the
    first observation, times[0] > 0 and the launch must anchor at t=0."""
    if times[0] == 0.0:
        return simulate(p0, v0, omega, times)
    return simulate(p0, v0, omega, np.concatenate([[0.0], times]))[1:]


def multicam_fit(times, frames_idx, uv, cams, p0, box, noise):
    sc, sa, t_hat, n_hat = noise

    def resid(theta):
        xyz = simulate_at(p0, theta[:3], theta[3:], times)
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
    cx, cy = float(d["K"][0, 2]), float(d["K"][1, 2])

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
            K = np.array([[fs[k], 0, cx], [0, fs[k], cy], [0, 0, 1.0]])
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


POLY_DEG = 5


def bundle_adjust(poses_path, track_path, first, last, fps, half,
                  out_poses, clip_id):
    """Joint fit: camera poses (smooth tripod curves) + trajectory, against
    ALL goal-corner observations and the ball track simultaneously. The
    ball and the corners choose the pose curves together — this removes the
    pose-systematic that smoothing-then-fitting leaves behind."""
    from src.goal_pnp import GOAL_3D

    d = np.load(poses_path)
    frames_all = d["frames"].astype(float)
    ok = d["ok"].astype(bool)
    corners_obs = d["corners"]
    C = d["C"]
    rv_raw = d["rvecs"].reshape(len(frames_all), 3)
    cx, cy = float(d["K"][0, 2]), float(d["K"][1, 2])

    # ball observations (refined centers); the kick may precede the first
    # observation (see the fit command's --kick-frame) — the prior fit's
    # json records it and simulate_at anchors the launch at t=0
    fb, uvb = refined_centers(track_path, first, last)
    fb = np.array(fb, float)
    kick = json.load(open(f"data/fits/{clip_id}.json")).get(
        "kick_frame", fb[0])
    times = (fb - kick) / fps
    sc, sa, t_hat, n_hat = estimate_track_noise(uvb)
    sc, sa = max(1.0, sc), max(2.0, sa)
    print(f"ball whitening: cross {sc:.2f}, along {sa:.2f} px")

    # goal geometry sign, as in smooth_poses
    obj = np.array([v for v in GOAL_3D.values()], float)
    best = None
    for s in (1.0, -1.0):
        o = obj.copy(); o[:, 0] *= s
        R0, _ = cv2.Rodrigues(rv_raw[np.where(ok)[0][0]])
        K0 = np.array([[d["fs"][np.where(ok)[0][0]], 0, cx],
                       [0, d["fs"][np.where(ok)[0][0]], cy], [0, 0, 1]])
        q = (K0 @ (R0 @ (o - C).T)).T
        uv = q[:, :2] / q[:, 2:]
        e = np.sqrt(np.mean((uv - corners_obs[np.where(ok)[0][0]]) ** 2))
        if best is None or e < best[1]:
            best = (o, e)
    obj_s = best[0]

    # initial pose curves + initial trajectory from the existing fit
    tnorm = (frames_all - frames_all.mean()) / 30.0
    coefR0 = [np.polyfit(tnorm[ok], rv_raw[ok, c], POLY_DEG) for c in range(3)]
    coefF0 = np.polyfit(tnorm[ok], d["fs"][ok], POLY_DEG)
    fit0 = json.load(open(f"data/fits/{clip_id}.json"))
    theta0 = np.array(fit0["theta6"])
    p0_init = np.array(fit0["p0"])
    box = tuple(fit0["box"])
    nC = POLY_DEG + 1

    def unpack_params(p):
        cR = [p[c * nC:(c + 1) * nC] for c in range(3)]
        cF = p[3 * nC:4 * nC]
        x0, y0 = p[4 * nC], p[4 * nC + 1]
        theta = p[4 * nC + 2:]
        return cR, cF, np.array([x0, y0, RADIUS]), theta

    def pose_at(cR, cF, tn):
        rv = np.array([np.polyval(cR[c], tn) for c in range(3)])
        R, _ = cv2.Rodrigues(rv)
        f = np.polyval(cF, tn)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        return K, R

    tn_all = tnorm
    tn_ball = (fb - frames_all.mean()) / 30.0

    def resid(p):
        cR, cF, p0, theta = unpack_params(p)
        r = []
        for k in np.where(ok)[0]:
            K, R = pose_at(cR, cF, tn_all[k])
            q = (K @ (R @ (obj_s - C).T)).T
            uv = q[:, :2] / q[:, 2:]
            r.append(((uv - corners_obs[k]) / 3.0).ravel())
        xyz = simulate_at(p0, theta[:3], theta[3:], times)
        duv = []
        for k in range(len(fb)):
            K, R = pose_at(cR, cF, tn_ball[k])
            q = K @ (R @ (xyz[k] - C))
            duv.append(q[:2] / q[2] - uvb[k])
        duv = np.array(duv)
        r.append(np.sum(duv * n_hat, axis=1) / sc)
        r.append(np.sum(duv * t_hat, axis=1) / sa)
        r.append(crossing_hinge(p0, theta[:3], theta[3:], box))
        r.append((np.array([p0[0], p0[1]]) - p0_init[:2]) / 0.3)
        return np.concatenate(r)

    p_init = np.concatenate(coefR0 + [coefF0, p0_init[:2], theta0])
    lo = np.concatenate([np.full(4 * nC, -np.inf), p0_init[:2] - 2.0, LOWER6])
    hi = np.concatenate([np.full(4 * nC, np.inf), p0_init[:2] + 2.0, UPPER6])
    print(f"bundle adjustment: {len(p_init)} params, "
          f"{ok.sum() * 8 + 2 * len(fb) + 4} residuals")
    res = least_squares(resid, p_init, bounds=(lo, hi), method="trf",
                        x_scale="jac")
    cR, cF, p0, theta = unpack_params(res.x)

    # report: corner rms and ball rms through the BA poses
    r = resid(res.x)
    n_corner = ok.sum() * 8
    corner_rms = float(np.sqrt(np.mean(r[:n_corner] ** 2)) * 3.0)
    cross_rms = float(np.sqrt(np.mean(r[n_corner:n_corner + len(fb)] ** 2)) * sc)
    along_rms = float(np.sqrt(np.mean(
        r[n_corner + len(fb):n_corner + 2 * len(fb)] ** 2)) * sa)
    print(f"BA: status {res.status}; corner rms {corner_rms:.2f} px, ball "
          f"cross rms {cross_rms:.2f} px, along rms {along_rms:.2f} px")

    # write BA poses npz for the renderer
    rvecs, tvecs, fs = [], [], []
    for k in range(len(frames_all)):
        K, R = pose_at(cR, cF, tn_all[k])
        rv, _ = cv2.Rodrigues(R)
        rvecs.append(rv.reshape(3, 1))
        tvecs.append((-R @ C).reshape(3, 1))
        fs.append(K[0, 0])
    np.savez(out_poses, K=d["K"], frames=d["frames"],
             rvecs=np.stack(rvecs), tvecs=np.stack(tvecs),
             fs=np.array(fs), C=C, ok=d["ok"], corners=corners_obs)
    print(f"BA poses -> {out_poses}")
    return theta, p0, box, (sc, sa, t_hat, n_hat), fb, uvb, times, kick


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smooth-poses")
    s.add_argument("poses"), s.add_argument("--out", required=True)
    b = sub.add_parser("ba")
    b.add_argument("clip_id"), b.add_argument("track"), b.add_argument("poses")
    b.add_argument("first", type=int), b.add_argument("last", type=int)
    b.add_argument("--fps", type=float, required=True)
    b.add_argument("--half", type=float, default=0.6)
    b.add_argument("--n-boot", type=int, default=25)
    b.add_argument("--out-poses", required=True)
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
    f.add_argument("--kick-frame", type=float, default=None,
                   help="frame of ball-ground contact when the kick is "
                        "before the first observation (may be negative "
                        "and fractional - contact is rarely on a frame)")
    f.add_argument("--launch-xy", type=float, nargs=2, default=None,
                   metavar=("X", "Y"),
                   help="measured launch point [m] (e.g. from the resting-"
                        "ball pixel), overrides the first-observation ray")
    f.add_argument("--noise", type=float, nargs=2, default=None,
                   metavar=("CROSS", "ALONG"),
                   help="explicit track noise [px] - the SG estimator "
                        "needs dense even spacing and inflates wildly on "
                        "sparse gap-heavy tracks")
    f.add_argument("--box", type=float, nargs=2, default=None,
                   metavar=("X", "Z"),
                   help="crossing box center [m], overriding pixel back-"
                        "projection (needed when the camera is oblique to "
                        "the goal plane: a pixel short of the plane back-"
                        "projects meters wide)")
    r = sub.add_parser("render")
    r.add_argument("fit_json"), r.add_argument("video")
    r.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "smooth-poses":
        smooth_poses(args.poses, args.out)
        return

    if args.cmd == "ba":
        theta, p0, box, noise, fb, uvb, times, kick = bundle_adjust(
            args.poses, args.track, args.first, args.last, args.fps,
            args.half, args.out_poses, args.clip_id)
        cams = load_cameras(args.out_poses)
        frames_idx = [int(f) for f in fb]
        xyz_fit = simulate_at(p0, theta[:3], theta[3:], times)
        uv_clean = np.array([project_frame(cams, f, xyz_fit[k])[0]
                             for k, f in enumerate(frames_idx)])
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(args.n_boot):
            uvb2 = anisotropic_noise(uv_clean, noise[0], noise[1], rng)
            rb = multicam_fit(times, frames_idx, uvb2, cams, p0, box, noise)
            if rb.status > 0:
                samples.append(rb.x)
        q = np.array([flight_quantities(sm) for sm in samples])
        lo, hi = np.percentile(q, [16, 84], axis=0)
        centre, half_i = 0.5 * (lo + hi), 0.5 * (hi - lo)
        bias, infl = spin_correction(noise[0])
        centre[3] -= bias
        half_i[3] *= infl
        half_i[4] *= SPIN_INTERVAL_INFLATION
        intervals = np.stack([centre - half_i, centre + half_i], axis=1)
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
            miss = np.hypot(fc[0] - box[0], fc[1] - box[1])
            print(f"\ngoal-mouth check: fitted x={fc[0]:+.2f}, z={fc[1]:.2f} "
                  f"vs measured x={box[0]:+.2f}, z={box[1]:.2f} -> {miss:.2f} m")
        out = f"data/fits/{args.clip_id}.json"
        json.dump({"clip_id": args.clip_id, "theta6": theta.tolist(),
                   "p0": p0.tolist(),
                   "window": [frames_idx[0], frames_idx[-1]],
                   "fps": args.fps, "box": list(box),
                   "quantities": dict(zip(QUANTITY_NAMES, qf.tolist())),
                   "intervals": intervals.tolist(),
                   "noise": [noise[0], noise[1]], "kick_frame": kick,
                   "samples": [sm.tolist() for sm in samples],
                   "poses": args.out_poses, "track": args.track,
                   "method": "bundle-adjusted"},
                  open(out, "w"), indent=1)
        print(f"saved {out}")
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
        t0f = args.kick_frame if args.kick_frame is not None else frames_idx[0]
        times = np.array([(f - t0f) / args.fps for f in frames_idx])
        print(f"{len(frames_idx)} usable flight observations"
              + (f", kick at frame {t0f}" if t0f != frames_idx[0] else ""))

        if args.launch_xy:
            p0 = np.array([args.launch_xy[0], args.launch_xy[1], RADIUS])
            print(f"launch point (measured): ({p0[0]:+.2f}, {p0[1]:.2f}, "
                  f"{p0[2]:.2f}) -> {np.hypot(p0[0], p0[1]):.1f} m out")
        else:
            p0 = ground_ray_point(cams, frames_idx[0], uv[0])
            print(f"launch point (ground ray): ({p0[0]:+.2f}, {p0[1]:.2f}, "
                  f"{p0[2]:.2f}) -> {np.hypot(p0[0], p0[1]):.1f} m out")

        if args.geom_crossing:
            ct = crossing_frame_geometric(cams, rows)
            cx, cz = goal_plane_point(cams, ct[0], (ct[1], ct[2]))
            print(f"measured crossing (goal-mouth entry, f{ct[0]}): "
                  f"x={cx:+.2f}, z={cz:.2f} +/- {args.half} m")
        elif args.box:
            cx, cz = args.box
            print(f"crossing box (measured entry): x={cx:+.2f}, z={cz:.2f} "
                  f"+/- {args.half} m")
        else:
            cx, cz = goal_plane_point(cams, frames_idx[-1], uv[-1])
            print(f"measured crossing (last tracked pixel, f{frames_idx[-1]}): "
                  f"x={cx:+.2f}, z={cz:.2f} +/- {args.half} m")
        box = (cx, cz, args.half)

        sc, sa, t_hat, n_hat = estimate_track_noise(uv)
        if args.noise:
            sc, sa = args.noise
        sc = float(np.hypot(sc, args.pose_noise))
        sa = float(np.hypot(sa, args.pose_noise))
        noise = (sc, sa, t_hat, n_hat)
        print(f"track noise incl. pose floor {args.pose_noise} px: "
              f"cross {sc:.2f} px, along {sa:.2f} px")

        res = multicam_fit(times, frames_idx, uv, cams, p0, box, noise)
        theta = res.x
        xyz_fit = simulate_at(p0, theta[:3], theta[3:], times)
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
                   "kick_frame": t0f,
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
    t0f = fit.get("kick_frame", a)   # kick may precede the first observation
    t_end = (b - t0f) / fps
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
            t = (idx - t0f) / fps
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
