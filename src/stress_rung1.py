"""Rung-1 stress test for per-frame goal PnP (tripod model), messi_live.

Four independent checks, all per-frame:
  1. leave-one-out: pose from 3 goal corners, reproject the held-out 4th
  2. out-of-solve landmark: corner-flag base (-34, 0, 0), template-tracked
     in the image vs projected through the solved pose
  3. seed-jitter sensitivity: +/-2 px noise on all corner seeds, spread of
     the projected penalty spot
  4. focal (zoom) curve smoothness

Run from repo root:  python -m src.stress_rung1
Writes reports/rung1_stress.png and reports/overlays/messi_live_flag_check.mp4
"""

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from src.goal_pnp import GOAL_3D, read_frames, track_corners, solve_poses

VIDEO = "data/raw_clips/messi_live.mp4"
POSES = "data/calibrations/messi_live_poses.npz"
SEED_FRAME = 8
SEEDS = {"near_top": (401, 478), "near_base": (412, 588),
         "far_top": (230, 560), "far_base": (228, 658)}
FLAG_SEED, FLAG_3D = (964, 357), (-34.0, 0.0, 0.0)
SPOT_3D = (0.0, 11.0, 0.0)
FIRST_GOOD = 5  # frames before the shot cut are junk


def tripod_solve(img_pts, obj_pts, C, p0):
    def resid(p):
        R, _ = cv2.Rodrigues(p[:3])
        K = np.array([[p[3], 0, 960], [0, p[3], 540], [0, 0, 1.0]])
        q = (K @ R @ (obj_pts - C).T).T
        return ((q[:, :2] / q[:, 2:]) - img_pts).ravel()
    return least_squares(resid, p0, method="lm").x


def project(p, C, pts3):
    R, _ = cv2.Rodrigues(p[:3])
    K = np.array([[p[3], 0, 960], [0, p[3], 540], [0, 0, 1.0]])
    q = (K @ R @ (np.atleast_2d(pts3) - C).T).T
    return q[:, :2] / q[:, 2:]


def main():
    d = np.load(POSES)
    C = d["C"]
    frames_idx = d["frames"]
    frames = read_frames(VIDEO)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    obj_all = None  # recover the solved side assignment from stored corners
    corners = track_corners(frames, SEED_FRAME, SEEDS)
    # side: reproject with both, pick the one matching stored solution
    for side in (1.0, -1.0):
        cand = np.array([(side * GOAL_3D[n][0],) + GOAL_3D[n][1:]
                         for n in GOAL_3D])
        p8 = np.concatenate([d["rvecs"][list(frames_idx).index(SEED_FRAME)]
                             .ravel(), [d["fs"][list(frames_idx).index(SEED_FRAME)]]])
        pr = project(p8, C, cand)
        img8 = np.array([corners[SEED_FRAME][n][:2] for n in GOAL_3D])
        if np.sqrt(np.mean((pr - img8) ** 2)) < 20:
            obj_all = cand
            side_used = side
            break
    assert obj_all is not None
    # the flag sits past the NEAR (image-right) post; near post x = side*3.66,
    # so the corner on that side is x = side*34
    flag3d = np.array([(side_used * 34.0, 0.0, 0.0)])

    # per-frame solved params from npz
    P = {int(f): np.concatenate([d["rvecs"][k].ravel(), [d["fs"][k]]])
         for k, f in enumerate(frames_idx)}

    # --- 1. leave-one-out ---
    names = list(GOAL_3D)
    loo = {n: [] for n in names}
    for i in sorted(P):
        if i < FIRST_GOOD:
            continue
        c = corners[i]
        for h, held in enumerate(names):
            keep = [j for j in range(4) if j != h]
            img3 = np.array([c[names[j]][:2] for j in keep], float)
            p = tripod_solve(img3, obj_all[keep], C, P[i])
            pr = project(p, C, obj_all[h])[0]
            loo[held].append(np.hypot(pr[0] - c[held][0], pr[1] - c[held][1]))

    # --- 2. corner-flag landmark ---
    tpl = gray[SEED_FRAME][FLAG_SEED[1]-12:FLAG_SEED[1]+13,
                           FLAG_SEED[0]-12:FLAG_SEED[0]+13]
    flag_err, flag_track = {}, {}
    prev = FLAG_SEED
    for i in sorted(P):
        if i < FIRST_GOOD:
            continue
        pu, pv = int(prev[0]), int(prev[1])
        y0, x0 = max(0, pv - 40), max(0, pu - 40)
        win = gray[i][y0:pv + 41, x0:pu + 41]
        if win.shape[0] < 27 or win.shape[1] < 27:
            continue
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        if score < 0.5:
            continue
        tr = (x0 + loc[0] + 12, y0 + loc[1] + 12)
        prev = tr
        pr = project(P[i], C, flag3d)[0]
        flag_track[i] = (tr, pr)
        flag_err[i] = float(np.hypot(pr[0] - tr[0], pr[1] - tr[1]))

    # --- 3. seed jitter ---
    rng = np.random.default_rng(0)
    spot_tracks = []
    for _ in range(7):
        jseeds = {n: (u + rng.normal(0, 2), v + rng.normal(0, 2))
                  for n, (u, v) in SEEDS.items()}
        jc = track_corners(frames, SEED_FRAME, jseeds)
        try:
            _, jp = solve_poses(jc, 4453.6, SEED_FRAME, (1920, 1080))
        except RuntimeError:
            continue
        spot_tracks.append({i: project(
            np.concatenate([jp[i][0].ravel(), [jp[i][3]]]), jp[i][2],
            np.array([SPOT_3D]))[0] for i in sorted(jp) if i >= FIRST_GOOD})
    common = sorted(set.intersection(*[set(t) for t in spot_tracks]))
    spread = [float(np.mean(np.std([t[i] for t in spot_tracks], axis=0)))
              for i in common]

    # --- figure ---
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    xs = [i for i in sorted(P) if i >= FIRST_GOOD]
    for n in names:
        axes[0].plot(xs, loo[n], label=n.replace("_", " "))
    axes[0].set_title("leave-one-out corner reprojection [px]")
    axes[0].legend(fontsize=7)
    axes[1].plot(sorted(flag_err), [flag_err[i] for i in sorted(flag_err)],
                 "o-", ms=3)
    axes[1].set_title("corner flag (out-of-solve, 30 m away) error [px]")
    axes[2].plot(xs, [P[i][3] for i in xs])
    axes[2].set_title("solved focal per frame [px] (zoom)")
    axes[3].plot(common, spread)
    axes[3].set_title("penalty-spot spread under 2 px seed jitter [px]")
    for ax in axes:
        ax.set_xlabel("frame")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/rung1_stress.png", dpi=130)

    # --- flag check video ---
    out = cv2.VideoWriter("reports/overlays/messi_live_flag_check.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
    for i in sorted(P):
        img = frames[i].copy()
        if i in flag_track:
            (tu, tv), (pu, pv) = flag_track[i]
            cv2.drawMarker(img, (int(tu), int(tv)), (0, 255, 0),
                           cv2.MARKER_TILTED_CROSS, 30, 2)   # tracked truth
            cv2.drawMarker(img, (int(pu), int(pv)), (0, 255, 255),
                           cv2.MARKER_CROSS, 30, 2)          # projected
            cv2.putText(img, f"flag err {flag_err[i]:.1f}px", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        out.write(img)
    out.release()

    print(f"leave-one-out median [px]: " + ", ".join(
        f"{n} {np.median(loo[n]):.1f}" for n in names))
    print(f"corner-flag error: median {np.median(list(flag_err.values())):.1f}, "
          f"p90 {np.percentile(list(flag_err.values()), 90):.1f} px "
          f"({len(flag_err)} frames tracked)")
    print(f"penalty-spot jitter spread: median {np.median(spread):.1f}, "
          f"max {np.max(spread):.1f} px ({len(spot_tracks)} jitter runs)")
    print("wrote reports/rung1_stress.png and "
          "reports/overlays/messi_live_flag_check.mp4")


if __name__ == "__main__":
    main()
