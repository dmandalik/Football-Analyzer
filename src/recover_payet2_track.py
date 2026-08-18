"""Ball track for payet2 by pan-aligned background subtraction.

SAM 2 failed twice on this 480p clip (static-lock, then shirt-snap) and
the old trim started 17 frames after the kick. The new trim (payet2,
source 92-140) starts pre-kick; this script detects the ball as the
moving blob against a per-frame median background, warped through the
goal-corner homography (exact under the tripod model), chained with a
velocity gate exactly like the messi2 recovery.

Kick: contact ~source 97 = f5 (ball at rest through f4, streak by f6).
Rest pixel (622, 358) in the pre-pan camera.

Run from repo root:  python -m src.recover_payet2_track
Writes data/tracks/payet2_recovered.csv, reports/payet2_track_tiles.png
and reports/overlays/payet2_ball.mp4 (watch before trusting).
"""

import csv
import os

import cv2
import numpy as np

from src.tracking import ballistic_check

POSES = "data/calibrations/payet2_poses.npz"
FRAMES = "data/raw_clips/payet2_frames"
N, W, H = 49, 854, 480
FPS = 30.0
REST = (622.0, 358.0)
KICK_F = 5.0          # refined against streak kinematics after first pass
FIRST, LAST = 6, 36   # search window: first streak .. goal arrival
SCORE_FLOOR = 10.0


def main():
    d = np.load(POSES)
    corners = d["corners"].astype(np.float32)
    fi = list(d["frames"])
    imgs = {f: cv2.imread(f"{FRAMES}/{f}.jpg") for f in range(N)}

    def homog(a, b):
        return cv2.getPerspectiveTransform(corners[fi.index(a)],
                                           corners[fi.index(b)])

    def bg_diff(f):
        """Warped triple-diff: fast movers only. A 2-frame gap clears the
        ball's own footprint; wall players and keeper move too slowly to
        survive the min(), unlike a median background they pollute."""
        a = cv2.warpPerspective(imgs[max(0, f - 2)], homog(max(0, f - 2), f),
                                (W, H)).astype(np.float32)
        c = cv2.warpPerspective(imgs[min(N - 1, f + 2)],
                                homog(min(N - 1, f + 2), f),
                                (W, H)).astype(np.float32)
        b = imgs[f].astype(np.float32)
        m = np.minimum(np.abs(b - a).mean(2), np.abs(b - c).mean(2))
        return cv2.GaussianBlur(m, (5, 5), 1.2)

    def best_blob(m, pred, r):
        """Nearest ball-sized motion blob to the prediction."""
        x0, y0 = int(pred[0]) - r, int(pred[1]) - r
        win = m[max(0, y0):y0 + 2 * r + 1, max(0, x0):x0 + 2 * r + 1]
        if win.size == 0:
            return None, 0.0
        th = (win > max(SCORE_FLOOR, 0.35 * win.max())).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(th)
        best = None
        for i in range(1, n):
            if not 4 <= stats[i, cv2.CC_STAT_AREA] <= 150:
                continue
            cu, cv_ = cent[i]
            dist = np.hypot(cu - (pred[0] - max(0, x0)),
                            cv_ - (pred[1] - max(0, y0)))
            score = float(win[lab == i].max())
            if best is None or dist < best[2]:
                best = ((max(0, x0) + cu, max(0, y0) + cv_), score, dist)
        if best is None:
            return None, float(win.max())
        return np.array(best[0]), best[1]

    # chain: start at the rest spot mapped into the first flight frame,
    # wide first window (launch direction unknown), then velocity-gated
    q = homog(4, FIRST) @ np.array([REST[0], REST[1], 1.0])
    pos = np.array([q[0] / q[2], q[1] / q[2]])   # chaining state, every frame
    vel = None
    chain = {}                                   # accepted observations only
    for f in range(FIRST, LAST + 1):
        if vel is None:
            pred, r = pos, 45          # first streak: anywhere around spot
        else:
            shift = homog(f - 1, f) @ np.array([*pos, 1.0])
            pred = np.array([shift[0] / shift[2], shift[1] / shift[2]]) + vel
            r = 16
        m = bg_diff(f)
        found, mx = best_blob(m, pred, r)
        hit = found is not None and mx > SCORE_FLOOR
        if hit:
            if vel is not None:
                shift = homog(f - 1, f) @ np.array([*pos, 1.0])
                base = np.array([shift[0] / shift[2], shift[1] / shift[2]])
                vel = 0.55 * vel + 0.45 * (found - base)
            else:
                vel = (found - pos) / max(1, f - FIRST)
            chain[f], pos = found, found
        else:
            pos = pred                 # coast through the miss
        print(f"f{f}: score {mx:5.1f} at ({pos[0]:6.1f},{pos[1]:6.1f})"
              f" {'HIT' if hit else 'coast'}"
              + (f"  vel ({vel[0]:+5.1f},{vel[1]:+5.1f})" if vel is not None
                 else ""))

    rows = [(f, chain[f][0], chain[f][1], 0, 1) for f in sorted(chain)]
    os.makedirs("data/tracks", exist_ok=True)
    with open("data/tracks/payet2_recovered.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "u", "v", "area", "ok"])
        w.writerows(rows)
    print(f"{len(rows)} detections -> data/tracks/payet2_recovered.csv")

    # guards in goal-anchored coords
    c0 = lambda f: corners[fi.index(f)][0]
    uv_g = [(u - c0(f)[0] + c0(0)[0], v - c0(f)[1] + c0(0)[1])
            for f, u, v, _, _ in rows]
    ok, msg = ballistic_check([r[0] for r in rows], uv_g, FPS)
    print(("PASS: " if ok else "FAIL: ") + msg)

    # verification tiles (zoomed crop at every detection) + overlay video
    tiles = []
    for f, u, v, _, _ in rows:
        t = imgs[f][max(0, int(v) - 24):int(v) + 25,
                    max(0, int(u) - 24):int(u) + 25]
        t = cv2.resize(t, (147, 147), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(t, (73, 73), (0, 0, 255), cv2.MARKER_CROSS, 26, 1)
        cv2.putText(t, str(f), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        tiles.append(t)
    rows_of_7 = [np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)]
    wmax = max(r.shape[1] for r in rows_of_7)
    rows_of_7 = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)))
                 for r in rows_of_7]
    cv2.imwrite("reports/payet2_track_tiles.png", np.vstack(rows_of_7))

    out = cv2.VideoWriter("reports/overlays/payet2_ball.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    trail = []
    by_f = {r[0]: r for r in rows}
    for f in range(N):
        img = imgs[f].copy()
        if f in by_f:
            trail.append((f, by_f[f][1], by_f[f][2]))
            cv2.circle(img, (int(by_f[f][1]), int(by_f[f][2])), 10,
                       (255, 0, 255), 2)
        pts = []
        for (g, u, v) in trail:
            qq = homog(g, f) @ np.array([u, v, 1.0])
            pts.append((int(qq[0] / qq[2]), int(qq[1] / qq[2])))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (0, 255, 255), 1)
        out.write(img)
    out.release()
    print("tiles -> reports/payet2_track_tiles.png; overlay -> "
          "reports/overlays/payet2_ball.mp4")


if __name__ == "__main__":
    main()
