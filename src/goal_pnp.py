"""Per-frame camera pose from the goal frame — no stabilization chain.

The goal is a rigid 7.32 x 2.44 m object visible through the flight, so
each frame's camera pose comes directly from PnP on its four corners
(post bases + crossbar junctions), template-tracked from a hand-seeded
reference frame. Focal is FIXED from a verified wide-frame calibration
(PnLCalib) — the goal alone is planar and can go focal-degenerate when
it faces the camera.

Every corner is matched against its ORIGINAL seed template each frame
(no drift accumulation). Frames where any corner match is weak keep the
previous pose and are flagged.

CLI (from repo root):
  python -m src.goal_pnp solve <video> <focal> --seed-frame N \
      --near-top U V --near-base U V --far-top U V --far-base U V \
      [--out npz]
  python -m src.goal_pnp markers <video> <poses.npz> [--out mp4]
      # deliverable-1 test: static pitch features must stay locked
"""

import argparse

import cv2
import numpy as np

GOAL_3D = {  # our frame: goal centre origin, +y into pitch, +z up
    "near_top": (3.66, 0.0, 2.44), "near_base": (3.66, 0.0, 0.0),
    "far_top": (-3.66, 0.0, 2.44), "far_base": (-3.66, 0.0, 0.0),
}
STATIC_MARKERS = {  # features NOT in the solve — the lock test
    "penalty spot": (0.0, 11.0, 0.0),
    "6yd corner +": (9.16, 5.5, 0.0),
    "6yd corner -": (-9.16, 5.5, 0.0),
}
TPL, SEARCH = 12, 34   # template half-size, search half-window [px]


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def track_corners(frames, seed_frame, seeds):
    """Two-stage corner tracking. Stage 1 matches the WHOLE goal as one
    structure-rich patch (thin white posts alone suffer the aperture
    problem — a corner template can slide along a bar with high NCC score,
    which is exactly how the wireframe walked off the goal on a panning
    clip). Stage 2 refines each corner locally around the global prior.
    Returns {frame: {name: (u, v, score)}}."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    ref = gray[seed_frame]
    h, w = ref.shape
    tpls = {}
    for name, (u, v) in seeds.items():
        u, v = int(u), int(v)
        tpls[name] = ref[v - TPL:v + TPL + 1, u - TPL:u + TPL + 1]
    us = [int(seeds[n][0]) for n in seeds]
    vs = [int(seeds[n][1]) for n in seeds]
    M = 25
    gx0, gy0 = max(0, min(us) - M), max(0, min(vs) - M)
    gx1, gy1 = min(w, max(us) + M), min(h, max(vs) + M)
    goal_tpl = ref[gy0:gy1, gx0:gx1]

    out = {}
    order = list(range(seed_frame, len(frames))) + \
            list(range(seed_frame - 1, -1, -1))
    prev = dict(seeds)
    prev_shift = (0.0, 0.0)
    for i in order:
        if i == seed_frame - 1:      # starting the backward leg
            prev, prev_shift = dict(seeds), (0.0, 0.0)
        # stage 1: global goal patch, generous search around previous shift
        sx0 = max(0, gx0 + int(prev_shift[0]) - 60)
        sy0 = max(0, gy0 + int(prev_shift[1]) - 60)
        sx1 = min(w, gx1 + int(prev_shift[0]) + 60)
        sy1 = min(h, gy1 + int(prev_shift[1]) + 60)
        win = gray[i][sy0:sy1, sx0:sx1]
        shift = prev_shift
        if win.shape[0] > goal_tpl.shape[0] and win.shape[1] > goal_tpl.shape[1]:
            r = cv2.matchTemplate(win, goal_tpl, cv2.TM_CCOEFF_NORMED)
            _, gs, _, gloc = cv2.minMaxLoc(r)
            if gs > 0.3:
                shift = (sx0 + gloc[0] - gx0, sy0 + gloc[1] - gy0)
        # stage 2: local corner refinement around seed + global shift
        cur = {}
        for name, tpl in tpls.items():
            pu = int(seeds[name][0] + shift[0])
            pv = int(seeds[name][1] + shift[1])
            y0, y1 = max(0, pv - 12), pv + 13
            x0, x1 = max(0, pu - 12), pu + 13
            win = gray[i][y0 - TPL:y1 + TPL, x0 - TPL:x1 + TPL] \
                if y0 - TPL >= 0 and x0 - TPL >= 0 else None
            if win is None or win.shape[0] < 2 * TPL + 5 or win.shape[1] < 2 * TPL + 5:
                cur[name] = (pu, pv, 0.0)
                continue
            r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            cur[name] = (x0 - TPL + loc[0] + TPL, y0 - TPL + loc[1] + TPL,
                         float(score))
        out[i] = cur
        prev = {n: (c[0], c[1]) for n, c in cur.items()}
        prev_shift = shift
    return out


def solve_poses(corners_by_frame, focal, seed_frame, image_size=(1920, 1080),
                min_score=0.45):
    """Fixed-K per-frame PnP, seeded at seed_frame and chained outward.

    Planar PnP has a two-fold mirror ambiguity; at the seed both IPPE
    solutions are computed and the one with the camera IN FRONT of the
    goal (C_y > 0, C_z > 0) is kept. Later frames refine iteratively from
    the neighbouring frame's pose, which locks the branch.
    """
    K = np.array([[focal, 0, image_size[0] / 2],
                  [0, focal, image_size[1] / 2], [0, 0, 1.0]])

    # which image-side post is x=+3.66 depends on which side of the pitch
    # the camera sits — try both assignments, keep the branch that puts the
    # camera in front of the goal and above the ground
    seed_rt, obj = None, None
    for side in (1.0, -1.0):
        cand = np.array([(side * GOAL_3D[n][0],) + GOAL_3D[n][1:]
                         for n in GOAL_3D], np.float32)
        c = corners_by_frame[seed_frame]
        img = np.array([[c[n][0], c[n][1]] for n in GOAL_3D], np.float32)
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            cand.reshape(-1, 1, 3), img.reshape(-1, 1, 2), K, None,
            flags=cv2.SOLVEPNP_IPPE)
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            C = (-R.T @ tv).ravel()
            if C[1] > 5 and C[2] > 1:
                seed_rt, obj = (rv, tv), cand
                print(f"post-side assignment: image-seeded 'near' post at "
                      f"x={side * 3.66:+.2f}; camera C=({C[0]:+.1f}, "
                      f"{C[1]:+.1f}, {C[2]:+.1f})")
                break
        if seed_rt:
            break
    if seed_rt is None:
        raise RuntimeError("no physical PnP branch (camera in front, above "
                           "ground) at the seed frame — degenerate view?")

    def img_pts(i):
        c = corners_by_frame[i]
        return np.array([[c[n][0], c[n][1]] for n in GOAL_3D], np.float32)

    # TRIPOD MODEL: a broadcast gantry camera pans/tilts/ZOOMS about a
    # fixed position. Fixed-focal PnP converts real zoom into fake camera
    # translation (z was drifting 22 -> 91 m). So: camera centre fixed from
    # the seed frame; per-frame free parameters are rotation and focal.
    from scipy.optimize import least_squares as _lsq

    rv0, tv0 = seed_rt
    R0, _ = cv2.Rodrigues(rv0)
    C_fixed = (-R0.T @ tv0).ravel()
    cx, cy = image_size[0] / 2, image_size[1] / 2

    def resid(p, img):
        rv, f = p[:3], p[3]
        R, _ = cv2.Rodrigues(rv)
        Kf = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        q = (Kf @ R @ (obj.astype(float) - C_fixed).T).T
        return ((q[:, :2] / q[:, 2:]) - img).ravel()

    poses = {}
    order = ([seed_frame]
             + list(range(seed_frame + 1, max(corners_by_frame) + 1))
             + list(range(seed_frame - 1, min(corners_by_frame) - 1, -1)))
    p = np.concatenate([rv0.ravel(), [focal]])
    p_seed = None
    for i in order:
        if i == seed_frame - 1 and p_seed is not None:
            p = p_seed.copy()             # restart chain for backward leg
        c = corners_by_frame[i]
        ok_match = all(c[n][2] > min_score for n in GOAL_3D)
        sol = _lsq(resid, p, args=(img_pts(i).astype(float),), method="lm")
        p = sol.x
        if i == seed_frame:
            p_seed = p.copy()
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        rvec = p[:3].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        tvec = (-R @ C_fixed).reshape(3, 1)
        poses[i] = (rvec.copy(), tvec.copy(), C_fixed.copy(), float(p[3]),
                    bool(ok_match and rms < 6.0))
    return K, poses


def project(K, rvec, tvec, pts3):
    uv, _ = cv2.projectPoints(np.asarray(pts3, np.float32), rvec, tvec, K, None)
    return uv.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("video"), s.add_argument("focal", type=float)
    s.add_argument("--seed-frame", type=int, required=True)
    for n in GOAL_3D:
        s.add_argument(f"--{n.replace('_', '-')}", type=float, nargs=2,
                       required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("markers")
    m.add_argument("video"), m.add_argument("poses")
    m.add_argument("--out", required=True)
    w = sub.add_parser("wireframe")
    w.add_argument("video"), w.add_argument("poses")
    w.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "wireframe":
        d = np.load(args.poses)
        frames = read_frames(args.video)
        h0, w0 = frames[0].shape[:2]
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w0, h0))
        gw, gh = 3.66, 2.44
        segs = [[(-gw, 0, 0), (-gw, 0, gh)], [(gw, 0, 0), (gw, 0, gh)],
                [(-gw, 0, gh), (gw, 0, gh)],
                [(-12, 0, 0), (12, 0, 0)]]          # goal line for context
        for k, i in enumerate(d["frames"]):
            img = frames[i].copy()
            col = (0, 255, 255) if d["ok"][k] else (0, 0, 255)
            f_k = float(d["fs"][k])
            Kk = np.array([[f_k, 0, w0 / 2], [0, f_k, h0 / 2], [0, 0, 1.0]])
            for a, b in segs:
                pa = project(Kk, d["rvecs"][k], d["tvecs"][k], [a])[0]
                pb = project(Kk, d["rvecs"][k], d["tvecs"][k], [b])[0]
                cv2.line(img, tuple(np.int32(pa)), tuple(np.int32(pb)), col, 2)
            cv2.putText(img, f"f{int(i)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(img)
        out.release()
        print(f"wireframe video -> {args.out}")
        return

    if args.cmd == "solve":
        frames = read_frames(args.video)
        seeds = {n: tuple(getattr(args, n)) for n in GOAL_3D}
        corners = track_corners(frames, args.seed_frame, seeds)
        h, w = frames[0].shape[:2]
        K, poses = solve_poses(corners, args.focal, args.seed_frame, (w, h))
        bad = [i for i in poses if not poses[i][4]]
        fs = np.array([poses[i][3] for i in sorted(poses)])
        print(f"{len(poses)} frames solved, {len(bad)} flagged"
              + (f" {sorted(bad)}" if bad else ""))
        print(f"tripod camera C=({poses[args.seed_frame][2][0]:+.1f}, "
              f"{poses[args.seed_frame][2][1]:+.1f}, "
              f"{poses[args.seed_frame][2][2]:+.1f}); focal range "
              f"{fs.min():.0f}..{fs.max():.0f} px (zoom)")
        np.savez(args.out, K=K,
                 frames=np.array(sorted(poses)),
                 rvecs=np.stack([poses[i][0] for i in sorted(poses)]),
                 tvecs=np.stack([poses[i][1] for i in sorted(poses)]),
                 fs=fs,
                 C=poses[args.seed_frame][2],
                 ok=np.array([poses[i][4] for i in sorted(poses)]),
                 corners=np.array([[corners[i][n][:2] for n in GOAL_3D]
                                   for i in sorted(poses)]))
    else:
        d = np.load(args.poses)
        K = d["K"]
        frames = read_frames(args.video)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frames[0].shape[:2]
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        h0, w0 = frames[0].shape[:2]
        for k, i in enumerate(d["frames"]):
            img = frames[i].copy()
            col = (0, 255, 255) if d["ok"][k] else (0, 0, 255)
            f_k = float(d["fs"][k]) if "fs" in d else float(K[0, 0])
            Kk = np.array([[f_k, 0, w0 / 2], [0, f_k, h0 / 2], [0, 0, 1.0]])
            for name, p3 in STATIC_MARKERS.items():
                u, v = project(Kk, d["rvecs"][k], d["tvecs"][k], [p3])[0]
                u, v = int(u), int(v)
                cv2.drawMarker(img, (u, v), col, cv2.MARKER_CROSS, 26, 2)
                cv2.putText(img, name, (u + 10, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            cv2.putText(img, f"f{int(i)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out.write(img)
        out.release()
        print(f"marker video -> {args.out}")


if __name__ == "__main__":
    main()
