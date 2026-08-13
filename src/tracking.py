"""SAM 2 click-to-propagate ball tracking, plus overlay rendering.

Usage (from repo root):
  python -m src.tracking track <video.mp4> <u> <v> [--frame N] [--out csv]
  python -m src.tracking overlay <video.mp4> <track.csv> [--out mp4]

`track` needs torch + sam2 and a checkpoint at MODEL_CHECKPOINT; it seeds
SAM 2 with one click on the ball and propagates through the clip, writing
frame,u,v,area,ok per line. `overlay` needs only cv2: it draws the track
on the footage so every track can be watched before it is trusted.
"""

import argparse
import csv
import os

import numpy as np

MODEL_CHECKPOINT = "models/sam2.1_hiera_small.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def extract_frames(video_path):
    """Dump a clip to the JPEG-per-frame directory SAM 2 reads natively
    (its mp4 loader needs decord, which has no Apple Silicon wheel)."""
    import subprocess

    frames_dir = os.path.splitext(video_path)[0] + "_frames"
    if not os.path.isdir(frames_dir) or not os.listdir(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video_path,
                        "-q:v", "2", "-start_number", "0",
                        os.path.join(frames_dir, "%d.jpg")], check=True)
    return frames_dir


def track_video(video_path, click_uv, click_frame=0, extra_points=None):
    """Propagate clicks on the ball through the whole clip. extra_points is
    a list of (u, v, frame) anchors on OTHER frames — use them when a fast
    blurred ball shares pixels with crisp static objects (boards, posts):
    a single-frame seed lets SAM 2 trade the streak for the static object,
    multi-frame anchors pin the moving one.
    Returns rows of (frame_idx, u, v, mask_area_px, ok)."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_CHECKPOINT,
                                           device=device)
    state = predictor.init_state(extract_frames(video_path))
    predictor.add_new_points_or_box(
        state, frame_idx=click_frame, obj_id=1,
        points=np.array([click_uv], dtype=np.float32),
        labels=np.array([1], dtype=np.int32))
    for (u, v, f) in (extra_points or []):
        predictor.add_new_points_or_box(
            state, frame_idx=int(f), obj_id=1,
            points=np.array([(u, v)], dtype=np.float32),
            labels=np.array([1], dtype=np.int32))

    # propagate both directions so a mid-flight seed covers the whole clip
    # (a pre-strike seed on a long-static ball tends to stay stuck to the
    # kick spot when the ball launches — seed mid-flight instead)
    by_frame = {}
    for reverse in (False, True):
        for frame_idx, _, masks in predictor.propagate_in_video(state,
                                                                reverse=reverse):
            mask = (masks[0] > 0.0).cpu().numpy().squeeze()
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                by_frame[frame_idx] = (frame_idx, np.nan, np.nan, 0, 0,
                                       np.nan, np.nan, np.nan, np.nan)
            else:
                pts = np.stack([xs, ys], 1).astype(float)
                c = pts.mean(0)
                # principal-axis extremes of the mask: a motion-blurred ball
                # is a streak, and its endpoints are physically meaningful
                # (start/end of exposure) where the area centroid wobbles
                # with mask-extent flicker
                if len(pts) > 4:
                    _, _, V = np.linalg.svd(pts - c, full_matrices=False)
                    proj = (pts - c) @ V[0]
                    lo, hi = pts[np.argmin(proj)], pts[np.argmax(proj)]
                else:
                    lo = hi = c
                by_frame[frame_idx] = (frame_idx, float(c[0]), float(c[1]),
                                       int(len(xs)), 1, float(lo[0]),
                                       float(lo[1]), float(hi[0]), float(hi[1]))
    rows = [by_frame[k] for k in sorted(by_frame)]

    # a ball mask is a few hundred px; a huge one means the click missed
    # the ball and SAM grabbed pitch/crowd — flag it loudly
    areas = [r[3] for r in rows if r[4]]  # chronological
    med = sorted(areas)[len(areas) // 2] if areas else 0
    if med > 10000:
        print(f"WARNING: median mask area {med} px — seed click likely "
              f"missed the ball; re-check the click point")
    # blur legitimately swells the mask ~2-3x with speed; a lock onto a
    # sock or pitch line survives the ceiling check but shows spikes or
    # sustained drift far outside that envelope
    if med:
        spiky = sum(1 for a in areas if a > 4 * med or a < med / 4)
        if spiky > 0.1 * len(areas):
            print(f"WARNING: {spiky}/{len(areas)} frames have mask area "
                  f">4x or <1/4x the median — mask is unstable; verify the "
                  f"overlay before trusting this track")
    return rows


def save_track(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ["frame", "u", "v", "area", "ok"]
    if rows and len(rows[0]) > 5:
        header += ["ax_lo_u", "ax_lo_v", "ax_hi_u", "ax_hi_v"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_track(path):
    with open(path) as f:
        return [(int(r["frame"]), float(r["u"]), float(r["v"]),
                 int(r["area"]), int(r["ok"]))
                for r in csv.DictReader(f)]


def render_overlay(video_path, track_path, out_path):
    """Draw the tracked ball position and its trail on every frame."""
    import cv2

    rows = {r[0]: r for r in load_track(track_path)}
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    trail = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = rows.get(idx)
        if row and row[4]:
            trail.append((int(row[1]), int(row[2])))
            cv2.circle(frame, trail[-1], 12, (0, 0, 255), 2)
        elif row:
            cv2.putText(frame, "LOST", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, a, b, (0, 255, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    print(f"overlay written to {out_path} ({idx} frames)")


def ballistic_check(frames, uv, fps):
    """Physics guard for a flight track: vertical image motion must show
    roughly constant downward curvature (gravity). Returns (ok, message).

    Catches the silent failure where a 'track' follows a static spot, a
    walking person, or camera pan — those are flat or non-curving. Run on
    the flight window only, in stabilized coordinates if the camera moves.
    """
    t = (np.asarray(frames, float) - frames[0]) / fps
    v = np.asarray(uv, float)[:, 1]
    if len(t) < 12:
        return False, f"only {len(t)} points — too few to verify a flight"
    coef, res, *_ = np.polyfit(t, v, 2, full=True)
    a = coef[0]  # px/s^2; image v grows downward, so gravity makes a > 0
    rms = float(np.sqrt(res[0] / len(t))) if len(res) else 0.0
    dv = v.max() - v.min()
    if dv < 30:
        return False, f"vertical motion only {dv:.0f} px — flat track, not a flight"
    if a < 50:
        return False, f"curvature {a:.0f} px/s^2 not downward-parabolic (want >>0)"
    half = len(t) // 2
    a1 = np.polyfit(t[:half], v[:half], 2)[0]
    a2 = np.polyfit(t[half:], v[half:], 2)[0]
    # both halves must curve downward; magnitude may legitimately drop
    # several-fold as the ball recedes (pixel scale shrinks with depth),
    # so the constancy bound is loose — the SIGN is the hard criterion
    if not (a1 > 0 and a2 > 0 and max(a1, a2) / max(min(a1, a2), 1e-9) < 8):
        return False, (f"curvature not roughly constant: halves "
                       f"{a1:.0f} / {a2:.0f} px/s^2")
    return True, (f"ballistic: curvature {a:.0f} px/s^2 (halves {a1:.0f}/"
                  f"{a2:.0f}), v-range {dv:.0f} px, quad rms {rms:.1f} px")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("video"), t.add_argument("u", type=float)
    t.add_argument("v", type=float)
    t.add_argument("--frame", type=int, default=0)
    t.add_argument("--point", type=float, nargs=3, action="append",
                   default=[], metavar=("U", "V", "FRAME"),
                   help="extra ball anchor on another frame (repeatable)")
    t.add_argument("--out", default=None)
    o = sub.add_parser("overlay")
    o.add_argument("video"), o.add_argument("track")
    o.add_argument("--out", default=None)
    c = sub.add_parser("check")
    c.add_argument("track"), c.add_argument("first", type=int)
    c.add_argument("last", type=int)
    c.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    if args.cmd == "check":
        rows = [r for r in load_track(args.track)
                if r[4] and args.first <= r[0] <= args.last]
        ok, msg = ballistic_check([r[0] for r in rows],
                                  [(r[1], r[2]) for r in rows], args.fps)
        print(("PASS: " if ok else "FAIL: ") + msg)
        return

    stem = os.path.splitext(os.path.basename(args.video))[0]
    if args.cmd == "track":
        rows = track_video(args.video, (args.u, args.v), args.frame,
                           extra_points=args.point)
        out = args.out or f"data/tracks/{stem}.csv"
        save_track(rows, out)
        ok = sum(r[4] for r in rows)
        print(f"{ok}/{len(rows)} frames tracked -> {out}")
    else:
        out = args.out or f"reports/overlays/{stem}_overlay.mp4"
        render_overlay(args.video, args.track, out)


if __name__ == "__main__":
    main()
