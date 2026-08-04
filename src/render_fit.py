"""Render a fitted 3D trajectory back onto the broadcast footage.

Draws, per frame: the bootstrap sample trajectories (thin, the uncertainty
bundle), the best-fit trajectory (thick), the ball's fitted position at
that frame time (circle), and the measured crossing box on the goal plane.

Run from repo root:
  python -m src.render_fit data/fits/<clip>.json <video.mp4> [--out mp4]
"""

import argparse
import json
import os

import cv2
import numpy as np

from src.fit_clip import load_camera
from src.physics import simulate


def project_poly(camera, xyz):
    uv = camera.project(xyz)
    return [tuple(np.int32(p)) for p in uv]


def draw_polyline(img, pts, color, thickness):
    for p, q in zip(pts, pts[1:]):
        cv2.line(img, p, q, color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_json")
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stab", default=None,
                    help="stabilization npz: draw in reference coords, warp "
                         "into each frame (panning cameras)")
    args = ap.parse_args()

    fit = json.load(open(args.fit_json))
    camera = load_camera(fit["calib"])
    theta = np.array(fit["theta6"])
    slomo = len(theta) > 6
    s = theta[6] if slomo else 1.0
    p0 = np.array(fit["p0"])
    a, b = fit["window"]
    fps = fit["fps"]
    t_end = (b - a) / fps / s          # true-time flight duration
    times_fine = np.linspace(0.0, t_end, 150)

    H_by_frame = None
    if args.stab:
        d = np.load(args.stab)
        H_by_frame = {int(f): d["H"][k] for k, f in enumerate(d["frames"])}

    best_pts = project_poly(camera, simulate(p0, theta[:3], theta[3:6], times_fine))
    sample_pts = [project_poly(camera, simulate(np.array(sp), np.array(st)[:3],
                                                np.array(st)[3:6], times_fine))
                  for st, sp in fit["samples"][:25]]

    cx, cz, half = fit["box"]
    box_pts = project_poly(camera, np.array(
        [[cx - half, 0, cz - half], [cx + half, 0, cz - half],
         [cx + half, 0, cz + half], [cx - half, 0, cz + half],
         [cx - half, 0, cz - half]]))

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = args.out or ("reports/overlays/"
                            + fit["clip_id"] + "_fit_overlay.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vfps = cap.get(cv2.CAP_PROP_FPS) or fps
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), vfps, (w, h))

    def warp(pts, Hf):
        out_pts = []
        for p in pts:
            q = Hf @ np.array([p[0], p[1], 1.0])
            out_pts.append(tuple(np.int32(q[:2] / q[2])))
        return out_pts

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        Hf = H_by_frame.get(idx) if H_by_frame else None
        bp = warp(best_pts, Hf) if Hf is not None else best_pts
        for pts in sample_pts:
            draw_polyline(frame, warp(pts, Hf) if Hf is not None else pts,
                          (180, 180, 60), 1)
        draw_polyline(frame, bp, (0, 200, 255), 2)
        draw_polyline(frame, warp(box_pts, Hf) if Hf is not None else box_pts,
                      (0, 0, 255), 2)
        t = (idx - a) / fps / s        # true time since launch
        if 0.0 <= t <= t_end:
            pos = simulate(p0, theta[:3], theta[3:6], np.array([0.0, t]))[-1]
            u, v = camera.project(np.array([pos]))[0]
            if Hf is not None:
                q = Hf @ np.array([u, v, 1.0])
                u, v = q[:2] / q[2]
            cv2.circle(frame, (int(u), int(v)), 10, (0, 200, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    print(f"fit overlay written to {out_path} ({idx} frames)")


if __name__ == "__main__":
    main()
