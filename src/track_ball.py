"""Generic ball tracking: learned detector + gravity-gated path search.

Phase C of the generalization plan: one command per clip, no per-clip
tuning. Candidates come from the WASB detector (finetuned checkpoint)
run tiled at native resolution with a LOW threshold; association is
dynamic programming over candidate edges with a near-constant-
acceleration cost in goal-anchored coordinates (per-frame corner
homographies, exact under the tripod model). Frames where no candidate
fits the physics stay EMPTY — the ODE fit bridges gaps honestly.

Usage (from repo root):
  python -m src.track_ball selftest
  python -m src.track_ball detect <clip_id> <frames_dir> <poses.npz> <first> <last>
  python -m src.track_ball associate <clip_id> <poses.npz>
detect writes data/tracks/<clip>_candidates.npz; associate writes
data/tracks/<clip>_auto.csv plus verification tiles.
"""

import csv
import os
import sys

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
TILE_W, TILE_H = 512, 288
K_PER_FRAME = 6          # candidates kept per frame
CONF_FLOOR = 0.10        # low threshold: over-generate, let physics choose
V_MAX = 90.0             # px/frame, goal-anchored
A_SIG = 3.0              # accel residual scale [px/frame^2]
G_PRIOR = (0.0, 0.9)     # rough image gravity; generous A_SIG absorbs error
MISS_COST = 14.0         # per skipped frame
CONF_W = 6.0             # weight of (1 - confidence)
MAX_GAP = 4
R_FRAME = 12.0       # reward per frame advanced: gap edges stay viable
                     # (net miss penalty = MISS_COST - R_FRAME per skipped frame)


def _homogs(poses_path):
    d = np.load(poses_path)
    corners = d["corners"].astype(np.float32)
    fi = list(d["frames"])

    def H(a, b):
        return cv2.getPerspectiveTransform(corners[fi.index(a)],
                                           corners[fi.index(b)])
    return H, fi


def _warp(Hm, u, v):
    q = Hm @ np.array([u, v, 1.0])
    return q[0] / q[2], q[1] / q[2]


# ---------------------------------------------------------------- detector
def detect(clip, frames_dir, poses_path, first, last,
           ckpt="models/wasb_ft.pth"):
    import torch
    sys.path.insert(0, "third_party/WASB-SBDT/src")
    from src.train_ball_detector import load_wasb
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_wasb(ckpt if os.path.exists(ckpt)
                      else "models/wasb_soccer_best.pth.tar").to(device)
    model.eval()

    out = {}
    for f in range(first, last + 1):
        imgs = [cv2.imread(f"{frames_dir}/{g}.jpg") for g in (f - 1, f, f + 1)]
        if any(im is None for im in imgs):
            continue
        ih, iw = imgs[0].shape[:2]
        xs = np.linspace(0, iw - TILE_W, max(1, round(iw / TILE_W * 1.25))
                         ).astype(int)
        ys = np.linspace(0, ih - TILE_H, max(1, round(ih / TILE_H * 1.25))
                         ).astype(int)
        cands = []
        for y0 in ys:
            for x0 in xs:
                stack = []
                for im in imgs:
                    c = im[y0:y0 + TILE_H, x0:x0 + TILE_W]
                    c = c.astype(np.float32)[:, :, ::-1] / 255.0
                    stack.append(((c - MEAN) / STD).transpose(2, 0, 1))
                x = torch.from_numpy(
                    np.concatenate(stack)[None].astype(np.float32)).to(device)
                with torch.no_grad():
                    hm = torch.sigmoid(model(x)[0][0, 1]).cpu()
                mp = torch.nn.functional.max_pool2d(hm[None, None], 5, 1, 2)[0, 0]
                pk = torch.nonzero((hm == mp) & (hm > CONF_FLOOR))
                for py, px in pk.tolist():
                    # peaks near a tile seam see a truncated ball; the
                    # overlapping neighbor tile covers that region properly
                    if ((px < 10 and x0 > 0) or (px > TILE_W - 10 and
                            x0 + TILE_W < iw) or (py < 10 and y0 > 0) or
                            (py > TILE_H - 10 and y0 + TILE_H < ih)):
                        continue
                    cands.append((x0 + px, y0 + py, hm[py, px].item()))
        # merge duplicates from tile overlap, keep top K
        cands.sort(key=lambda c: -c[2])
        kept = []
        for u, v, s in cands:
            if all(np.hypot(u - u2, v - v2) > 6 for u2, v2, _ in kept):
                kept.append((u, v, s))
            if len(kept) >= K_PER_FRAME:
                break
        out[f] = kept
        if (f - first) % 10 == 0:
            print(f"f{f}: {len(kept)} candidates "
                  f"(best {kept[0][2]:.2f})" if kept else f"f{f}: none",
                  flush=True)
    np.savez(f"data/tracks/{clip}_candidates.npz",
             frames=np.array(sorted(out)),
             cands=np.array([out[f] + [(np.nan,) * 3] * (K_PER_FRAME - len(out[f]))
                             for f in sorted(out)], dtype=np.float32))
    print(f"saved data/tracks/{clip}_candidates.npz "
          f"({len(out)} frames)")


# ------------------------------------------------------------- association
def viterbi(frames, cands_g, conf):
    """DP over edges (t1,i)->(t2,j) with near-constant-acceleration cost.
    cands_g: goal-anchored candidate positions per frame index.
    Returns {frame: cand_index} for the best physically consistent path."""
    edges = []          # (fa, ia, fb, ib, base_cost)
    for a, fa in enumerate(frames):
        for b in range(a + 1, min(a + 1 + MAX_GAP, len(frames))):
            fb = frames[b]
            dt = fb - fa
            if dt > MAX_GAP:
                break
            for i, pa in enumerate(cands_g[fa]):
                for j, pb in enumerate(cands_g[fb]):
                    if np.isnan(pa[0]) or np.isnan(pb[0]):
                        continue
                    vel = (np.array(pb) - pa) / dt
                    if np.linalg.norm(vel) > V_MAX or np.linalg.norm(vel) < 2:
                        continue
                    cost = (MISS_COST * (dt - 1)
                            + CONF_W * (1 - conf[fb][j]))
                    edges.append((fa, i, fb, j, cost))
    by_end = {}
    for k, e in enumerate(edges):
        by_end.setdefault((e[2], e[3]), []).append(k)
    dp = np.full(len(edges), np.inf)
    parent = np.full(len(edges), -1, int)
    g = np.array(G_PRIOR)
    for k, (fa, i, fb, j, base) in enumerate(edges):
        dp[k] = base - R_FRAME * (fb - fa)
        for k2 in by_end.get((fa, i), []):
            f0, i0 = edges[k2][0], edges[k2][1]
            p0 = cands_g[f0][i0]
            pa, pb = cands_g[fa][i], cands_g[fb][j]
            dt1, dt2 = fa - f0, fb - fa
            v1 = (np.array(pa) - p0) / dt1
            v2 = (np.array(pb) - pa) / dt2
            acc = (v2 - v1) / (0.5 * (dt1 + dt2))
            c = (dp[k2] + base - R_FRAME * (fb - fa)
                 + float(np.sum((acc - g) ** 2)) / A_SIG ** 2)
            if c < dp[k]:
                dp[k], parent[k] = c, k2
    # min total cost wins; rewards make long consistent paths negative
    best = int(np.argmin(dp)) if len(dp) else None
    path = {}
    kk = best
    while kk is not None and kk >= 0:
        fa, i, fb, j, _ = edges[kk]
        path[fb] = j
        path[fa] = i
        kk = parent[kk] if parent[kk] >= 0 else None
    return path


def associate(clip, poses_path, frames_dir=None):
    d = np.load(f"data/tracks/{clip}_candidates.npz")
    frames = [int(f) for f in d["frames"]]
    raw = d["cands"]
    H, _ = _homogs(poses_path)
    ref = frames[0]
    cands_g, conf, cands_px = {}, {}, {}
    for k, f in enumerate(frames):
        Hm = H(f, ref)
        cands_g[f] = [(_warp(Hm, u, v) if np.isfinite(u) else (np.nan, np.nan))
                      for u, v, s in raw[k]]
        conf[f] = [s for _, _, s in raw[k]]
        cands_px[f] = [(u, v) for u, v, s in raw[k]]
    path = viterbi(frames, cands_g, conf)
    rows = [(f, *cands_px[f][j], 0, 1) for f, j in sorted(path.items())]
    out = f"data/tracks/{clip}_auto.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "u", "v", "area", "ok"])
        w.writerows(rows)
    cov = len(rows) / len(frames) * 100
    print(f"{len(rows)}/{len(frames)} frames on path ({cov:.0f}%) -> {out}")
    if frames_dir:
        tiles = []
        for f, u, v, _, _ in rows[::max(1, len(rows) // 20)]:
            img = cv2.imread(f"{frames_dir}/{f}.jpg")
            t = img[max(0, int(v) - 22):int(v) + 23,
                    max(0, int(u) - 22):int(u) + 23]
            t = cv2.resize(t, (120, 120), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(t, (60, 60), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.putText(t, str(f), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, .4,
                        (0, 255, 255), 1)
            tiles.append(t)
        rows_i = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
        wmax = max(r.shape[1] for r in rows_i)
        rows_i = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)))
                  for r in rows_i]
        cv2.imwrite(f"reports/{clip}_auto_tiles.png", np.vstack(rows_i))
        print(f"tiles -> reports/{clip}_auto_tiles.png — LOOK at them")
    return rows


# ---------------------------------------------------------------- selftest
def selftest():
    """Synthetic: known parabola + decoys; the DP must recover the truth."""
    rng = np.random.default_rng(3)
    frames = list(range(60))
    truth = {}
    u, v, vu, vv = 200.0, 800.0, 22.0, -18.0
    for f in frames:
        truth[f] = (u, v)
        u, v, vv = u + vu, v + vv, vv + 0.9
    cands_g, conf = {}, {}
    dropped = set(rng.choice(frames[5:55], 6, replace=False))
    for f in frames:
        cs, ss = [], []
        if f not in dropped:                       # true ball, noisy
            cs.append((truth[f][0] + rng.normal(0, 1.2),
                       truth[f][1] + rng.normal(0, 1.2)))
            ss.append(0.5 + 0.4 * rng.random())
        for _ in range(4):                          # static + slow decoys
            base = rng.choice([(400, 700), (900, 300), (1300, 820)])
            cs.append((base[0] + rng.normal(0, 2) + 0.5 * f,
                       base[1] + rng.normal(0, 2)))
            ss.append(0.3 + 0.6 * rng.random())     # decoys can be confident
        while len(cs) < K_PER_FRAME:
            cs.append((np.nan, np.nan)), ss.append(0.0)
        cands_g[f], conf[f] = cs, ss
    path = viterbi(frames, cands_g, conf)
    hits = sum(1 for f, j in path.items() if j == 0 and f not in dropped)
    wrong = sum(1 for f, j in path.items()
                if (f in dropped) or (j != 0))
    print(f"selftest: path covers {len(path)}/60 frames, "
          f"true-candidate hits {hits}, wrong picks {wrong}")
    ok = hits >= 45 and wrong == 0
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "detect":
        detect(sys.argv[2], sys.argv[3], sys.argv[4],
               int(sys.argv[5]), int(sys.argv[6]),
               *(sys.argv[7:8] or []))
    elif cmd == "associate":
        associate(sys.argv[2], sys.argv[3],
                  sys.argv[4] if len(sys.argv) > 4 else None)
