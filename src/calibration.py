"""PnLCalib wrapper: single-frame broadcast camera calibration.

Uses the official PnLCalib implementation as a library (external/PnLCalib,
cloned from github.com/mguti97/PnLCalib; weights in models/) — their
working code, not a reimplementation. Their world frame is pitch-centred,
x along the pitch length, z NEGATIVE up (crossbar at z=-2.44). Output is
converted to this project's frame: target-goal centre origin, +y into the
pitch, +z up, right-handed.

Usage (from repo root):
  python -m src.calibration <image.png> --goal right
Writes data/calibrations/<stem>.json and a pitch-overlay render to
reports/overlays/<stem>_calib.png — look at it before trusting it.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image

sys.path.insert(0, "external/PnLCalib")

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
                                 get_keypoints_from_heatmap_batch_maxpool_l,
                                 complete_keypoints, coords_to_dict)

from src.synthetic import Camera

WEIGHTS_KP = "models/SV_kp"
WEIGHTS_LINE = "models/SV_lines"
KP_THRESHOLD = 0.3434
LINE_THRESHOLD = 0.7867
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0

_resize = T.Resize((540, 960))


def load_models(device="cpu"):
    cfg = yaml.safe_load(open("external/PnLCalib/config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open("external/PnLCalib/config/hrnetv2_w48_l.yaml"))
    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(WEIGHTS_KP, map_location=device))
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(WEIGHTS_LINE, map_location=device))
    for m in (model, model_l):
        m.to(device).eval()
    return model, model_l


def calibrate_image(image_path, device="cpu", pnl_refine=True):
    """Mirror of PnLCalib's inference() for a single image. Returns their
    final_params_dict (pitch-centred frame) plus the image size."""
    model, model_l = load_models(device)
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise FileNotFoundError(image_path)
    h_orig, w_orig = frame_bgr.shape[:2]

    frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    frame = f.to_tensor(frame).float().unsqueeze(0)
    if frame.size()[-1] != 960:
        frame = _resize(frame)
    frame = frame.to(device)
    _, _, h, w = frame.size()

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)
    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1])
    kp_dict = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
    lines_dict = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0],
                                             w=w, h=h, normalize=True)

    cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
    cam.update(kp_dict, lines_dict)
    params = cam.heuristic_voting(refine_lines=pnl_refine)
    if params is None:
        raise RuntimeError("PnLCalib could not calibrate this frame")
    return params, (w_orig, h_orig)


def goal_frame_axes(goal):
    """Origin and axis matrix M of our frame expressed in PnLCalib's
    pitch-centred frame: their_point = origin + M @ our_point."""
    if goal == "right":
        origin = np.array([PITCH_LENGTH / 2, 0.0, 0.0])
        M = np.array([[0.0, -1.0, 0.0],    # our x = their -y
                      [-1.0, 0.0, 0.0],    # our y = their -x
                      [0.0, 0.0, -1.0]]).T  # our z = their -z
    else:
        origin = np.array([-PITCH_LENGTH / 2, 0.0, 0.0])
        M = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]]).T
    return origin, M


def to_project_camera(params, goal):
    """PnLCalib cam params -> synthetic.Camera in our goal-centred frame."""
    cp = params["cam_params"]
    K = np.array([[cp["x_focal_length"], 0.0, cp["principal_point"][0]],
                  [0.0, cp["y_focal_length"], cp["principal_point"][1]],
                  [0.0, 0.0, 1.0]])
    R_their = np.array(cp["rotation_matrix"])
    C_their = np.array(cp["position_meters"])
    origin, M = goal_frame_axes(goal)
    C_ours = M.T @ (C_their - origin)
    R_ours = R_their @ M
    return Camera(K, R_ours, C_ours)


def compose_to_reference(camera, H):
    """Express a frame-f* calibration in REFERENCE coordinates, given the
    stabilization homography H mapping reference pixels -> frame-f* pixels
    (stabilize.py convention): P_ref = H^-1 P_f*, re-decomposed into a
    pinhole Camera via RQ."""
    from scipy.linalg import rq

    P = camera.K @ camera.R @ np.hstack([np.eye(3), -camera.C[:, None]])
    P_ref = np.linalg.inv(H) @ P
    M = P_ref[:, :3]
    K, R = rq(M)
    T = np.diag(np.sign(np.diag(K)))
    K, R = K @ T, T @ R
    C = -np.linalg.inv(M) @ P_ref[:, 3]
    return Camera(K / K[2, 2], R, C)


def render_pitch_overlay(image_path, camera, out_path):
    """Project our-frame pitch model onto the frame. Wrong calibration puts
    lines in the stands — always look."""
    gw, gh = 7.32 / 2, 2.44
    box_w, box_d, spot = 40.32 / 2, 16.5, 11.0
    segments = [
        [(-gw, 0, 0), (-gw, 0, gh)], [(gw, 0, 0), (gw, 0, gh)],
        [(-gw, 0, gh), (gw, 0, gh)],                        # goal frame
        [(-box_w, 0, 0), (-box_w, box_d, 0)],
        [(box_w, 0, 0), (box_w, box_d, 0)],
        [(-box_w, box_d, 0), (box_w, box_d, 0)],            # penalty box
        [(-34, 0, 0), (34, 0, 0)],                          # goal line
    ]
    img = cv2.imread(image_path)
    for a, b in segments:
        ia = camera.project(np.array([a], float))[0]
        ib = camera.project(np.array([b], float))[0]
        cv2.line(img, tuple(np.int32(ia)), tuple(np.int32(ib)), (255, 0, 0), 2)
    ps = camera.project(np.array([[0.0, spot, 0.0]]))[0]
    cv2.circle(img, tuple(np.int32(ps)), 6, (0, 0, 255), -1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--goal", choices=["left", "right"], default="right",
                    help="which goal (in PnLCalib pitch x) the kick targets")
    args = ap.parse_args()

    params, size = calibrate_image(args.image)
    camera = to_project_camera(params, args.goal)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    os.makedirs("data/calibrations", exist_ok=True)
    out_json = f"data/calibrations/{stem}.json"
    json.dump({"K": camera.K.tolist(), "R": camera.R.tolist(),
               "C": camera.C.tolist(), "goal": args.goal,
               "image_size": size}, open(out_json, "w"), indent=1)
    print(f"camera position (our frame): {np.round(camera.C, 2)} "
          f"(x along goal line, y into pitch, z up)")
    print(f"saved {out_json}")

    overlay = f"reports/overlays/{stem}_calib.png"
    render_pitch_overlay(args.image, camera, overlay)
    print(f"pitch overlay -> {overlay}")


if __name__ == "__main__":
    main()
