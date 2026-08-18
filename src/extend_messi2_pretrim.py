"""Add the pre-trim flight frames (source 491-494 = trim -4..-1) to the
messi2 track and poses.

The kick is at source 490; the trim starts at 495, so the first four
airborne frames — the ones that pin the launch direction — are missing
from the fit. The anchored fit shows an azimuth/spin degeneracy (initial
direction trades against curl), and these frames are the direct
measurement that resolves it.

Ball: median background over s470-489 (walking people vanish, the
resting ball is masked out), largest motion blob in the launch corridor,
PCA midpoint = mid-exposure position, matching SAM 2 streak centroids.
Poses: goal corners template-tracked back from trim f0, then the same
tripod solve as goal_pnp (fixed C, per-frame rotation + focal).

Run from repo root:  python -m src.extend_messi2_pretrim
Writes data/calibrations/messi2_poses_ext.npz,
data/tracks/messi2_recovered_ext.csv, reports/messi2_pretrim_check.png
"""

import csv

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.goal_pnp import GOAL_3D, track_corners

POSES = "data/calibrations/messi2_poses.npz"
TRACK = "data/tracks/messi2_recovered.csv"
SRC = "data/raw_clips/messi2_src.mp4"
TRIM0 = 495                      # trim f0 in source numbering
PRE = [491, 492, 493, 494]       # airborne pre-trim frames
BG = list(range(470, 490, 2))
CORRIDOR = ((770, 940), (330, 430))   # (u range, v range) launch corridor
REST_PX = (930, 404)
# streak centers verified by eye on grid-annotated diff maps; the blob
# detector merges the streak with pitch-line flicker on two frames (the
# camera starts panning at contact, so lines light up in the diff)
MANUAL = {491: (916.0, 392.5), 492: (879.0, 378.5),
          493: (848.5, 363.5), 494: (817.0, 350.0)}


def tripod_solve(img_pts, obj, C, K0, p_init):
    def resid(p):
        R, _ = cv2.Rodrigues(p[:3])
        K = np.array([[p[3], 0, K0[0, 2]], [0, p[3], K0[1, 2]], [0, 0, 1.0]])
        q = (K @ R @ (obj - C).T).T
        return ((q[:, :2] / q[:, 2:]) - img_pts).ravel()
    r = least_squares(resid, p_init, method="lm")
    return r.x, np.sqrt(np.mean(r.fun ** 2))


def main():
    d = np.load(POSES)
    fi = list(d["frames"])
    K0 = d["K"]
    C = d["C"]

    cap = cv2.VideoCapture(SRC)
    src = {}
    for i in range(TRIM0 + 1):
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"source ended at {i}")
        if i in BG or PRE[0] - 2 <= i <= TRIM0:
            src[i] = img
    cap.release()

    # corners: seed on s495 with the trim-f0 corners, track back
    span = sorted(f for f in src if f >= PRE[0] - 2)
    names = list(GOAL_3D)
    seeds = {n: tuple(c) for n, c in zip(names, d["corners"][fi.index(0)])}
    ctrk = track_corners([src[f] for f in span], span.index(TRIM0), seeds)

    # goal x-sign the original solve used (test both on trim f0's pose)
    k0 = fi.index(0)
    R0, _ = cv2.Rodrigues(d["rvecs"][k0])
    best = None
    for s in (1.0, -1.0):
        obj = np.array([(s * GOAL_3D[n][0],) + tuple(GOAL_3D[n][1:])
                        for n in names])
        K = np.array([[d["fs"][k0], 0, K0[0, 2]],
                      [0, d["fs"][k0], K0[1, 2]], [0, 0, 1.0]])
        q = (K @ R0 @ (obj - C).T).T
        e = np.sqrt(np.mean((q[:, :2] / q[:, 2:]
                             - d["corners"][k0]) ** 2))
        if best is None or e < best[1]:
            best = (obj, e)
    obj = best[0]

    p_init = np.concatenate([d["rvecs"][k0].ravel(), [d["fs"][k0]]])
    poses_new = {}
    for f in PRE:
        img_pts = np.array([ctrk[span.index(f)][n][:2] for n in names], float)
        p, rms = tripod_solve(img_pts, obj, C, K0, p_init)
        poses_new[f] = p
        print(f"s{f}: tripod solve corner rms {rms:.2f} px, focal {p[3]:.0f}")

    # ball streaks via median background, searched in tight per-frame
    # windows along the rest -> trim-f0 line (people moving through the
    # wide corridor otherwise out-blob the ball)
    bg = np.median(np.stack([src[f].astype(np.float32) for f in BG]),
                   axis=0)
    F0_PX = (787.2, 338.9)
    tiles = []
    ball = {}
    for f in PRE:
        frac = (f - (PRE[0] - 1)) / (TRIM0 - (PRE[0] - 1))
        pu = REST_PX[0] + frac * (F0_PX[0] - REST_PX[0])
        pv = REST_PX[1] + frac * (F0_PX[1] - REST_PX[1])
        diff = np.abs(src[f].astype(np.float32) - bg).mean(2)
        u0, v0 = int(pu) - 30, int(pv) - 22
        win = cv2.GaussianBlur(diff[v0:v0 + 45, u0:u0 + 61], (5, 5), 1.5)
        cv2.circle(win, (REST_PX[0] - u0, REST_PX[1] - v0), 12, 0, -1)
        m = (win > max(15.0, 0.4 * win.max())).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            print(f"s{f}: no streak found")
            continue
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(lab == big)
        c = np.stack([xs, ys], 1).astype(float).mean(0)
        auto = (u0 + c[0], v0 + c[1])
        ball[f] = MANUAL.get(f, auto)
        print(f"s{f}: streak center {ball[f]} (auto ({auto[0]:.1f}, "
              f"{auto[1]:.1f}), {len(xs)} px)")
        t = src[f][v0 - 20:v0 + 65, u0 - 30:u0 + 91].copy()
        cv2.drawMarker(t, (int(c[0]) + 30, int(c[1]) + 20), (0, 0, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.putText(t, f"s{f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        tiles.append(cv2.resize(t, (605, 425)))
    cv2.imwrite("reports/messi2_pretrim_check.png",
                np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))

    # extended poses npz: pre-trim frames get negative trim indices
    frames_ext = np.concatenate([[f - TRIM0 for f in PRE], d["frames"]])
    rvecs_ext = np.concatenate([[poses_new[f][:3].reshape(3, 1) for f in PRE],
                                d["rvecs"]])
    fs_ext = np.concatenate([[poses_new[f][3] for f in PRE], d["fs"]])
    tvecs_ext = np.concatenate([
        [(-cv2.Rodrigues(poses_new[f][:3])[0] @ C).reshape(3, 1)
         for f in PRE], d["tvecs"]])
    ok_ext = np.concatenate([[True] * len(PRE), d["ok"]])
    corners_ext = np.concatenate([
        [[ctrk[span.index(f)][n][:2] for n in names] for f in PRE],
        d["corners"]])
    np.savez("data/calibrations/messi2_poses_ext.npz", K=K0,
             frames=frames_ext, rvecs=rvecs_ext, tvecs=tvecs_ext,
             fs=fs_ext, C=C, ok=ok_ext, corners=corners_ext)

    rows = [(f - TRIM0, ball[f][0], ball[f][1], 0, 1) for f in PRE
            if f in ball]
    with open(TRACK) as fh:
        rows += [(int(r["frame"]), float(r["u"]), float(r["v"]),
                  int(r["area"]), int(r["ok"])) for r in csv.DictReader(fh)]
    with open("data/tracks/messi2_recovered_ext.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "u", "v", "area", "ok"])
        w.writerows(sorted(rows))
    print(f"extended: {len(rows)} rows -> data/tracks/"
          f"messi2_recovered_ext.csv; poses -> messi2_poses_ext.npz")
    print("LOOK at reports/messi2_pretrim_check.png before fitting")


if __name__ == "__main__":
    main()
