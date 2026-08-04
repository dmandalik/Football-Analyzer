"""Per-frame ECC stabilization to a reference frame — standard pipeline.

Broadcast cameras follow the ball, so tracks and calibrations live in
per-frame coordinates. This module computes one homography per frame
(ECC, warm-started frame to frame), maps tracked pixels into reference-
frame coordinates, and estimates the residual stabilization error by
template-matching static background patches — that error compounds over
long clips and belongs in the noise model, not discovered later.

Convention: H_i maps reference-frame pixel coords -> frame-i pixel coords
(cv2.findTransformECC with the reference as template). Tracked pixels in
frame i become reference coords via H_i^{-1}.

CLI (from repo root):
  python -m src.stabilize compute <video.mp4> [--ref 0] [--out npz]
  python -m src.stabilize apply <track.csv> <homographies.npz> [--out csv]
"""

import argparse
import os

import cv2
import numpy as np

from src.tracking import load_track, save_track


def read_gray(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def compute_homographies(frames, ref_idx=0):
    """{frame_idx: 3x3 H} for every frame; identity for the reference."""
    ref = frames[ref_idx]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    out = {ref_idx: np.eye(3)}
    warp = np.eye(3, dtype=np.float32)
    for i in range(ref_idx + 1, len(frames)):
        try:
            _, warp = cv2.findTransformECC(ref, frames[i], warp,
                                           cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        except cv2.error:
            print(f"  ECC failed at frame {i}; keeping previous warp")
        out[i] = warp.astype(np.float64).copy()
    warp = np.eye(3, dtype=np.float32)
    for i in range(ref_idx - 1, -1, -1):
        try:
            _, warp = cv2.findTransformECC(ref, frames[i], warp,
                                           cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        except cv2.error:
            print(f"  ECC failed at frame {i}; keeping previous warp")
        out[i] = warp.astype(np.float64).copy()
    return out


def to_reference(uv, H):
    """Map one frame-i pixel into reference coordinates."""
    p = np.linalg.inv(H) @ np.array([uv[0], uv[1], 1.0])
    return p[0] / p[2], p[1] / p[2]


def stabilization_error(frames, H_by_frame, ref_idx=0, n_patches=6):
    """Residual mis-registration per frame [px], from template-matching
    static high-contrast patches of the reference in each stabilized frame."""
    ref = frames[ref_idx]
    h, w = ref.shape
    pts = cv2.goodFeaturesToTrack(ref[: int(h * 0.55)], 60, 0.05, 80)
    pts = [tuple(np.int32(p[0])) for p in pts if
           40 < p[0][0] < w - 40 and 40 < p[0][1] < h * 0.5][:n_patches]
    errs = {}
    for i, H in sorted(H_by_frame.items()):
        stab = cv2.warpPerspective(frames[i], np.linalg.inv(H), (w, h))
        offs = []
        for (x, y) in pts:
            tpl = ref[y - 16:y + 16, x - 16:x + 16]
            win = stab[y - 26:y + 26, x - 26:x + 26]
            if tpl.size == 0 or win.shape[0] < 52 or win.shape[1] < 52:
                continue
            r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            if score > 0.5:
                offs.append(np.hypot(loc[0] - 10, loc[1] - 10))
        if offs:
            errs[i] = float(np.median(offs))
    return errs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("video"), c.add_argument("--ref", type=int, default=0)
    c.add_argument("--out", default=None)
    a = sub.add_parser("apply")
    a.add_argument("track"), a.add_argument("homographies")
    a.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "compute":
        frames = read_gray(args.video)
        print(f"{len(frames)} frames; computing ECC homographies...")
        H = compute_homographies(frames, args.ref)
        errs = stabilization_error(frames, H, args.ref)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out = args.out or f"data/calibrations/{stem}_stab.npz"
        np.savez(out, ref=args.ref,
                 frames=np.array(sorted(H)),
                 H=np.stack([H[k] for k in sorted(H)]),
                 err_frames=np.array(sorted(errs)),
                 err_px=np.array([errs[k] for k in sorted(errs)]))
        e = np.array([errs[k] for k in sorted(errs)])
        print(f"stabilization error: median {np.median(e):.2f} px, "
              f"p90 {np.percentile(e, 90):.2f} px, max {e.max():.2f} px")
        print(f"saved {out}")
    else:
        d = np.load(args.homographies)
        H_by_frame = {int(f): d["H"][k] for k, f in enumerate(d["frames"])}
        rows = []
        for r in load_track(args.track):
            if r[4] and r[0] in H_by_frame:
                u, v = to_reference((r[1], r[2]), H_by_frame[r[0]])
                rows.append((r[0], u, v, r[3], 1))
            else:
                rows.append((r[0], np.nan, np.nan, 0, 0))
        out = args.out or args.track.replace(".csv", "_stab.csv")
        save_track(rows, out)
        print(f"stabilized track -> {out}")


if __name__ == "__main__":
    main()
