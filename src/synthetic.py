"""Synthetic ground-truth generator: known trajectory -> 2D pixel track.

World frame: origin at the centre of the goal line on the ground, +x along the
goal line, +y into the pitch (toward the kicker), +z up. Goal mouth spans
x in [-3.66, 3.66], z in [0, 2.44] at y = 0.
"""

import numpy as np

from src.physics import simulate, time_to_plane_y0

GOAL_HALF_WIDTH = 7.32 / 2.0
GOAL_HEIGHT = 2.44


class Camera:
    """Pinhole camera: pixel = K @ R @ (X - C), then perspective divide."""

    def __init__(self, K, R, C):
        self.K = np.asarray(K, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.C = np.asarray(C, dtype=float)

    def project(self, points):
        """(N,3) world points -> (N,2) pixel coordinates."""
        points = np.atleast_2d(points)
        cam = (self.R @ (points - self.C).T).T
        uvw = (self.K @ cam.T).T
        return uvw[:, :2] / uvw[:, 2:3]


def look_at_camera(position, target, focal_px, image_size=(1920, 1080)):
    """Camera at `position` looking at `target`, world +z as up."""
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    K = np.array([
        [focal_px, 0.0, image_size[0] / 2.0],
        [0.0, focal_px, image_size[1] / 2.0],
        [0.0, 0.0, 1.0],
    ])
    return Camera(K, R, position)


def broadcast_camera():
    """Main-gantry-style camera: halfway line, beyond the touchline, elevated,
    zoomed on the goal area."""
    return look_at_camera(
        position=(-42.0, 50.0, 16.0),
        target=(0.0, 11.0, 1.5),
        focal_px=2800.0,
    )


def generate_track(p0, v0, omega, camera, fps=25.0, noise_px=2.0, seed=0):
    """Simulate a flight until it crosses the goal plane and observe it.

    Returns a dict with frame times, true 3D positions, clean pixel track,
    and the noisy pixel track (iid Gaussian noise, sigma = noise_px).
    """
    t_end = time_to_plane_y0(p0, v0, omega)
    if t_end is None:
        raise ValueError("trajectory never reaches the goal plane y=0")

    n_frames = int(np.floor(t_end * fps)) + 1
    times = np.arange(n_frames) / fps
    xyz = simulate(p0, v0, omega, times)
    uv_true = camera.project(xyz)

    rng = np.random.default_rng(seed)
    uv_noisy = uv_true + rng.normal(0.0, noise_px, size=uv_true.shape)
    return {
        "times": times,
        "xyz": xyz,
        "uv_true": uv_true,
        "uv_noisy": uv_noisy,
        "t_end": t_end,
    }
