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
    args = ap.parse_args()

    fit = json.load(open(args.fit_json))
    camera = load_camera(fit["calib"])
    theta = np.array(fit["theta6"])
    p0 = np.array(fit["p0"])
    a, b = fit["window"]
    fps = fit["fps"]
    t_end = (b - a) / fps
    times_fine = np.linspace(0.0, t_end, 150)

    best_pts = project_poly(camera, simulate(p0, theta[:3], theta[3:], times_fine))
    sample_pts = [project_poly(camera, simulate(np.array(sp), np.array(st)[:3],
                                                np.array(st)[3:], times_fine))
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

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for pts in sample_pts:
            draw_polyline(frame, pts, (180, 180, 60), 1)
        draw_polyline(frame, best_pts, (0, 200, 255), 2)
        draw_polyline(frame, box_pts, (0, 0, 255), 2)
        t = (idx - a) / fps
        if 0.0 <= t <= t_end:
            pos = simulate(p0, theta[:3], theta[3:], np.array([0.0, t]))[-1]
            u, v = camera.project(np.array([pos]))[0]
            cv2.circle(frame, (int(u), int(v)), 10, (0, 200, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    print(f"fit overlay written to {out_path} ({idx} frames)")


if __name__ == "__main__":
    main()
