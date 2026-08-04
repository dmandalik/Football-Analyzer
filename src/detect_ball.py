"""Classical ball detection for stabilized clips: median-background
subtraction in reference coordinates plus RANSAC trajectory consensus.

Fallback and cross-check for SAM 2, which under camera pan can lock onto
static spots, spare balls, or birds. Principles:
  - detections stay RAW: RANSAC only rejects outliers with a LOOSE
    threshold; it never reshapes accepted points — a real trajectory is
    not a parabola (drag and Magnus bend it), so a tight parabola gate
    would bias the physics fit;
  - report how much was rejected;
  - the caller must run tracking.ballistic_check and eyeball the marked
    frames before trusting the output.

CLI (from repo root):
  python -m src.detect_ball <video> <stab.npz> <first> <last> \
      [--anchor U V FRAME] [--tol 25] [--out csv]
"""

import argparse

import numpy as np
import cv2

from src.stabilize import read_gray
from src.tracking import ballistic_check, save_track


def stabilized_frames(frames, H_by_frame):
    h, w = frames[0].shape
    out = {}
    for i, H in H_by_frame.items():
        out[i] = cv2.warpPerspective(frames[i], np.linalg.inv(H), (w, h))
    return out


def candidates_per_frame(stab, first, last, thresh=18, max_cands=12):
    """Blob candidates per frame from median-background subtraction."""
    sample = [stab[i] for i in sorted(stab)][::max(1, len(stab) // 25)]
    bg = np.median(np.stack(sample), axis=0).astype(np.int16)
    cands = {}
    for i in range(first, last + 1):
        if i not in stab:
            continue
        d = np.abs(stab[i].astype(np.int16) - bg).astype(np.uint8)
        d[stab[i] == 0] = 0
        _, t = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
        n, _, stats, cents = cv2.connectedComponentsWithStats(t)
        cs = [(float(cents[j][0]), float(cents[j][1]), int(stats[j][4]))
              for j in range(1, n) if 8 < stats[j][4] < 1500]
        cs.sort(key=lambda c: -c[2])
        cands[i] = cs[:max_cands]
    return cands


def ransac_track(cands, tol=25.0, anchor=None, anchor2=None, iters=4000,
                 seed=0):
    """Consensus quadratics u(t), v(t) over per-frame candidates; returns
    the raw detections consistent with the best model. anchor2 is a known
    endpoint with an uncertain frame: (u, v, f_min, f_max)."""
    rng = np.random.default_rng(seed)
    frames = sorted(f for f in cands if cands[f])
    if anchor:
        au, av, af = anchor
    best_score, best_model = -1, None
    for _ in range(iters):
        pick = rng.choice(len(frames), size=3, replace=False)
        pts = []
        for k in pick:
            f = frames[k]
            c = cands[f][rng.integers(len(cands[f]))]
            pts.append((f, c[0], c[1]))
        if anchor:
            pts[0] = (af, au, av)
        if anchor2:
            u2, v2, f_lo, f_hi = anchor2
            pts[1] = (float(rng.integers(int(f_lo), int(f_hi) + 1)), u2, v2)
        ts = np.array([p[0] for p in pts], float)
        if len(set(ts)) < 3:
            continue
        cu = np.polyfit(ts, [p[1] for p in pts], 2)
        cv_ = np.polyfit(ts, [p[2] for p in pts], 2)
        score = 0
        for f in frames:
            ue, ve = np.polyval(cu, f), np.polyval(cv_, f)
            if any(np.hypot(c[0] - ue, c[1] - ve) < tol for c in cands[f]):
                score += 1
        if score > best_score:
            best_score, best_model = score, (cu, cv_)

    cu, cv_ = best_model
    # refit on inliers once (still only for SELECTION, not for output)
    for _ in range(2):
        ts, us, vs = [], [], []
        for f in frames:
            ue, ve = np.polyval(cu, f), np.polyval(cv_, f)
            near = min(cands[f], key=lambda c: np.hypot(c[0] - ue, c[1] - ve))
            if np.hypot(near[0] - ue, near[1] - ve) < tol:
                ts.append(f), us.append(near[0]), vs.append(near[1])
        cu, cv_ = np.polyfit(ts, us, 2), np.polyfit(ts, vs, 2)

    rows, rejected = [], 0
    for f in frames:
        ue, ve = np.polyval(cu, f), np.polyval(cv_, f)
        near = min(cands[f], key=lambda c: np.hypot(c[0] - ue, c[1] - ve))
        if np.hypot(near[0] - ue, near[1] - ve) < tol:
            rows.append((f, near[0], near[1], near[2], 1))
        else:
            rejected += 1
    return rows, rejected, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"), ap.add_argument("stab")
    ap.add_argument("first", type=int), ap.add_argument("last", type=int)
    ap.add_argument("--anchor", type=float, nargs=3, default=None,
                    metavar=("U", "V", "FRAME"))
    ap.add_argument("--anchor2", type=float, nargs=4, default=None,
                    metavar=("U", "V", "FMIN", "FMAX"))
    ap.add_argument("--tol", type=float, default=25.0)
    ap.add_argument("--thresh", type=int, default=18)
    ap.add_argument("--max-cands", type=int, default=12)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = read_gray(args.video)
    d = np.load(args.stab)
    H_by_frame = {int(f): d["H"][k] for k, f in enumerate(d["frames"])}
    stab = stabilized_frames(frames, H_by_frame)
    cands = candidates_per_frame(stab, args.first, args.last,
                                 thresh=args.thresh, max_cands=args.max_cands)
    n_cand = sum(len(v) for v in cands.values())
    print(f"{n_cand} candidates over {len(cands)} frames")

    anchor = tuple(args.anchor) if args.anchor else None
    anchor2 = tuple(args.anchor2) if args.anchor2 else None
    rows, rejected, total = ransac_track(cands, tol=args.tol, anchor=anchor,
                                         anchor2=anchor2)
    print(f"RANSAC (tol {args.tol} px): selected {len(rows)}/{total} frames, "
          f"rejected {rejected}")
    save_track(rows, args.out)
    ok, msg = ballistic_check([r[0] for r in rows],
                              [(r[1], r[2]) for r in rows], args.fps)
    print(("PASS: " if ok else "FAIL: ") + msg)
    print(f"saved {args.out} (REFERENCE-frame coordinates)")


if __name__ == "__main__":
    main()
