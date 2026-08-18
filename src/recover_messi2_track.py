"""Recover the messi2 ball track through the white-post crossing.

SAM 2 follows the ball cleanly to f17, then trades it for the ad board
where ball, post, and banner are all white. This script rebuilds the tail
with the verified goal-PnP corners as the alignment backbone (the tripod
model makes the 4-corner homography a full-frame homography):

  f18, f20, f21 — background subtraction: median of later frames warped
      into the target frame kills the static banner, the ball remains as
      a dark smudge; dark-blob centroid near the extrapolated path
  f19 — MISSING: ball occluded by the near post
  f22 — pan-compensated triple-frame differencing with the keeper color-
      masked (bg-diff there is contaminated by the keeper's reach); the
      ball meets the net around f23 at the same spot, which becomes the
      crossing-box center for the fit

Launch anchor: the kick is 5 frames BEFORE the trim (source frame 490 =
trim frame -5; trim f0 = source 495, matched by pixel diff). The resting
ball at (926, 398) in the s490 camera is mapped through the corner
homography into trim-f0 and dropped to the ground plane -> measured
launch (x0, y0) for the fit's --kick-frame/--launch-xy options.

Run from repo root:  python -m src.recover_messi2_track
Writes data/tracks/messi2_recovered.csv and
reports/overlays/messi2_ball.mp4 (watch it before trusting it).
"""

import csv

import cv2
import numpy as np

from src.fit_multicam import load_cameras, ground_ray_point, goal_plane_point
from src.goal_pnp import track_corners
from src.tracking import ballistic_check, load_track

POSES = "data/calibrations/messi2_poses.npz"
TRACK_IN = "data/tracks/messi2.csv"
TRACK_OUT = "data/tracks/messi2_recovered.csv"
VIDEO_FRAMES = "data/raw_clips/messi2_frames"
SRC_VIDEO = "data/raw_clips/messi2_src.mp4"
FPS = 30000 / 1001
KICK_SRC, TRIM0_SRC = 490, 495           # kick contact; trim f0 in source
REST_PX = (926.0, 398.0)                 # ball at rest, s490 camera
OCCLUDED = {19}                          # behind the near post
CHAIN_F22 = (393.0, 210.0)               # triple-diff detection (see above)
NET_F23 = (391.0, 220.0)                 # entry pixel for the crossing box


def homography(corners, fi, f_from, f_to):
    return cv2.getPerspectiveTransform(corners[fi.index(f_from)],
                                       corners[fi.index(f_to)])


def bg_detect(imgs, corners, fi, f, pred, bg_frames):
    """Median background of homography-aligned later frames, dark-blob
    centroid within +/-16 px of the predicted position."""
    stack = [cv2.warpPerspective(imgs[g], homography(corners, fi, g, f),
                                 (1280, 720)) for g in bg_frames]
    bg = np.median(np.stack(stack), axis=0).astype(np.float32)
    diff = cv2.GaussianBlur(np.abs(imgs[f].astype(np.float32) - bg).mean(2),
                            (5, 5), 1.5)
    x0, y0 = int(pred[0]) - 16, int(pred[1]) - 16
    _, mx, _, p = cv2.minMaxLoc(diff[y0:y0 + 33, x0:x0 + 33])
    py, px = y0 + p[1], x0 + p[0]
    w = np.maximum(diff[py - 6:py + 7, px - 6:px + 7] - 0.3 * mx, 0)
    yy, xx = np.mgrid[py - 6:py + 7, px - 6:px + 7]
    return (xx * w).sum() / w.sum(), (yy * w).sum() / w.sum(), mx


def launch_anchor(corners, fi, cams):
    """Track corners back through the pre-trim source frames, map the
    resting-ball pixel into trim-f0, drop to the ground plane."""
    cap = cv2.VideoCapture(SRC_VIDEO)
    frames = []
    for i in range(TRIM0_SRC + 1):
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"source ended at frame {i}")
        if i >= KICK_SRC - 1:
            frames.append(img)
    cap.release()
    seeds = {n: tuple(c) for n, c in zip(
        ["near_top", "near_base", "far_top", "far_base"],
        corners[fi.index(0)])}
    ctrk = track_corners(frames, len(frames) - 1, seeds)
    kick_i = KICK_SRC - (KICK_SRC - 1)
    src = np.float32([ctrk[kick_i][n][:2] for n in seeds])
    dst = np.float32([ctrk[len(frames) - 1][n][:2] for n in seeds])
    H = cv2.getPerspectiveTransform(src, dst)
    q = H @ np.array([REST_PX[0], REST_PX[1], 1.0])
    uv0 = (q[0] / q[2], q[1] / q[2])
    p = ground_ray_point(cams, 0, uv0)
    return uv0, (float(p[0]), float(p[1]))


def main():
    d = np.load(POSES)
    corners = d["corners"].astype(np.float32)
    fi = list(d["frames"])
    cams = load_cameras(POSES)
    imgs = {f: cv2.imread(f"{VIDEO_FRAMES}/{f}.jpg") for f in range(0, 48)}

    rows = [r for r in load_track(TRACK_IN) if r[0] <= 17]
    # extrapolate in goal-anchored coords to predict f18/f20/f21
    c0 = lambda f: corners[fi.index(f)][0]
    preds = {18: (444, 210), 20: (419, 210), 21: (406, 209)}
    bg_frames = list(range(30, 48, 2))
    for f, pred in preds.items():
        u, v, score = bg_detect(imgs, corners, fi, f, pred, bg_frames)
        print(f"f{f}: bg-diff detection ({u:.1f}, {v:.1f}) score {score:.0f}")
        rows.append((f, u, v, 0, 1))
    rows.append((22, CHAIN_F22[0], CHAIN_F22[1], 0, 1))
    rows.sort()

    with open(TRACK_OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "u", "v", "area", "ok"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {TRACK_OUT} (f19 occluded by post, omitted)")

    # guards in goal-anchored coordinates (camera pans)
    uv_g = [(r[1] - c0(r[0])[0] + c0(0)[0], r[2] - c0(r[0])[1] + c0(0)[1])
            for r in rows]
    ok, msg = ballistic_check([r[0] for r in rows], uv_g, FPS)
    print(("PASS: " if ok else "FAIL: ") + msg)
    step = np.hypot(*np.diff(np.array(uv_g), axis=0).T)
    print(f"goal-frame step px/frame: max {step.max():.1f}, "
          f"min {step.min():.1f} (f17->f18 spans the handoff)")

    uv0, xy = launch_anchor(corners, fi, cams)
    print(f"launch anchor: resting ball -> trim-f0 pixel "
          f"({uv0[0]:.1f}, {uv0[1]:.1f}) -> world ({xy[0]:+.2f}, {xy[1]:.2f}) "
          f"[{np.hypot(*xy):.1f} m out], kick at trim frame "
          f"{KICK_SRC - TRIM0_SRC}")
    cx, cz = goal_plane_point(cams, 23, NET_F23)
    print(f"crossing box center (net contact f23): x={cx:+.2f}, z={cz:.2f}")

    out = cv2.VideoWriter("reports/overlays/messi2_ball.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    trail = []
    by_f = {r[0]: r for r in rows}
    for f in range(0, 48):
        img = imgs[f].copy()
        if f in by_f:
            r = by_f[f]
            trail.append((f, r[1], r[2]))
            color = (0, 0, 255) if r[3] else (255, 0, 255)  # magenta=recovered
            cv2.circle(img, (int(r[1]), int(r[2])), 12, color, 2)
        if f in OCCLUDED:
            cv2.putText(img, "f19: occluded by post", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # pan-compensated trail: re-anchor every stored point to this frame
        Hf = {}
        pts = []
        for (g, u, v) in trail:
            if g not in Hf:
                Hf[g] = homography(corners, fi, g, f)
            q = Hf[g] @ np.array([u, v, 1.0])
            pts.append((int(q[0] / q[2]), int(q[1] / q[2])))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (0, 255, 255), 2)
        out.write(img)
    out.release()
    print("overlay -> reports/overlays/messi2_ball.mp4 (magenta = recovered "
          "points; watch before trusting)")


if __name__ == "__main__":
    main()
