"""Structure of real tracking residuals: anisotropy, bias signature,
autocorrelation.

Residuals are taken against a Savitzky-Golay smooth of the track and
rotated into along-track / cross-track components. Notes on identifiability:
a CONSTANT along-track pull (blur dragging the centroid toward the travel
direction) is absorbed by the smooth and cannot be seen without ground
truth — but a blur pull is proportional to image speed, so its observable
signatures are (a) along-track residual correlating with speed and
(b) mask area correlating with speed. Both are reported, as are lag-1/2
autocorrelations (SAM 2 mask drift would violate the iid assumption the
spin calibration is built on).

Run from repo root:  python -m src.experiment_track_noise
"""

import numpy as np
from scipy.signal import savgol_filter

from src.tracking import load_track

FLIGHT_WINDOWS = {"messi_kick": (8, 36), "ronaldo_kick": (30, 90),
                  "calhanoglu_kick": (25, 115)}


def analyze(name, window):
    rows = [r for r in load_track(f"data/tracks/{name}.csv")
            if r[4] and window[0] <= r[0] <= window[1]]
    uv = np.array([[r[1], r[2]] for r in rows])
    area = np.array([r[3] for r in rows], float)

    smooth = savgol_filter(uv, window_length=9, polyorder=2, axis=0)
    res = uv - smooth
    vel = np.gradient(smooth, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    t_hat = vel / np.maximum(speed[:, None], 1e-9)
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    along = np.sum(res * t_hat, axis=1)
    cross = np.sum(res * n_hat, axis=1)

    def ac(x, lag):
        x = x - x.mean()
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    print(f"\n{name} ({len(rows)} flight frames):")
    print(f"  sigma along {along.std():.2f} px, cross {cross.std():.2f} px "
          f"(ratio {along.std() / cross.std():.1f})")
    print(f"  mean along {along.mean():+.2f} px, cross {cross.mean():+.2f} px "
          f"(vs smooth; constant pull is unobservable by construction)")
    print(f"  autocorr along lag1 {ac(along, 1):+.2f} lag2 {ac(along, 2):+.2f}; "
          f"cross lag1 {ac(cross, 1):+.2f} lag2 {ac(cross, 2):+.2f}")
    slope = np.polyfit(speed, along, 1)[0]
    print(f"  blur-pull signature: d(along)/d(speed) {slope:+.3f} px per px/frame, "
          f"corr(area, speed) {np.corrcoef(area, speed)[0, 1]:+.2f}")


def main():
    for name, window in FLIGHT_WINDOWS.items():
        analyze(name, window)


if __name__ == "__main__":
    main()
