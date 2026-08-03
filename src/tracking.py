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


def track_video(video_path, click_uv, click_frame=0):
    """Propagate a single click on the ball through the whole clip.
    Returns rows of (frame_idx, u, v, mask_area_px, ok)."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = build_sam2_video_predictor(MODEL_CONFIG, MODEL_CHECKPOINT,
                                           device=device)
    state = predictor.init_state(video_path)
    predictor.add_new_points_or_box(
        state, frame_idx=click_frame, obj_id=1,
        points=np.array([click_uv], dtype=np.float32),
        labels=np.array([1], dtype=np.int32))

    rows = []
    for frame_idx, _, masks in predictor.propagate_in_video(state):
        mask = (masks[0] > 0.0).cpu().numpy().squeeze()
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            rows.append((frame_idx, np.nan, np.nan, 0, 0))
        else:
            rows.append((frame_idx, float(xs.mean()), float(ys.mean()),
                         int(len(xs)), 1))
    return sorted(rows)


def save_track(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "u", "v", "area", "ok"])
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("track")
    t.add_argument("video"), t.add_argument("u", type=float)
    t.add_argument("v", type=float)
    t.add_argument("--frame", type=int, default=0)
    t.add_argument("--out", default=None)
    o = sub.add_parser("overlay")
    o.add_argument("video"), o.add_argument("track")
    o.add_argument("--out", default=None)
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    if args.cmd == "track":
        rows = track_video(args.video, (args.u, args.v), args.frame)
        out = args.out or f"data/tracks/{stem}.csv"
        save_track(rows, out)
        ok = sum(r[4] for r in rows)
        print(f"{ok}/{len(rows)} frames tracked -> {out}")
    else:
        out = args.out or f"reports/overlays/{stem}_overlay.mp4"
        render_overlay(args.video, args.track, out)


if __name__ == "__main__":
    main()
