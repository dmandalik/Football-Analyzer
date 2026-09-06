"""Finetune the WASB soccer ball detector on our verified free-kick labels.

Interim step of the tracking-generalization plan (Phase B-lite): prove the
training loop moves the zero-shot baseline before the SoccerNet corpus
arrives. Leave-one-clip-out: ronaldo_wc is never trained on.

Samples: 512x288 crops from 3 consecutive frames with the ball placed at a
random position inside the crop (no center bias); targets are per-frame
Gaussian heatmaps from the track. 25% of samples are ball-free crops of the
same frames — hard negatives for posts/stewards/boards. Augmentation:
horizontal flip + brightness/contrast jitter.

Run from repo root:
  python -m src.train_ball_detector check   # dataset sanity: shapes + a png
  python -m src.train_ball_detector train   # finetune -> models/wasb_ft.pth
  python -m src.train_ball_detector eval [ckpt]  # top1/top5@5px per clip
"""

import csv
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "third_party/WASB-SBDT/src")
from omegaconf import OmegaConf          # noqa: E402
from models import build_model           # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
W, H, SIGMA = 512, 288, 2.5
CLIPS = {
    "messi_live": ("data/tracks/messi_live_pca.csv",
                   "data/raw_clips/messi_live_frames"),
    "messi2": ("data/tracks/messi2_recovered_ext.csv",
               "data/raw_clips/messi2_frames"),
    "ronaldo_wc": ("data/tracks/ronaldo_wc_recovered.csv",
                   "data/raw_clips/ronaldo_wc_frames"),
    "messi_kick": ("data/tracks/messi_kick.csv",
                   "data/raw_clips/messi_kick_frames"),
    "calhanoglu": ("data/tracks/calhanoglu_kick.csv",
                   "data/raw_clips/calhanoglu_kick_frames"),
}
HELD_OUT = "ronaldo_wc"


def load_wasb(ckpt="models/wasb_soccer_best.pth.tar"):
    cfg = OmegaConf.create({"model": OmegaConf.load(
        "third_party/WASB-SBDT/src/configs/model/wasb.yaml")})
    model = build_model(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    # MPS: BatchNorm backward crashes on non-contiguous inputs (the HRNet
    # multi-branch fuse produces them) - force contiguity going in
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.register_forward_pre_hook(
                lambda mod, inp: (inp[0].contiguous(),))
    return model


def tracks(clip):
    path, fdir = CLIPS[clip]
    t = {}
    for r in csv.DictReader(open(path)):
        if int(r.get("ok", "1")) == 1:
            t[int(r["frame"])] = (float(r["u"]), float(r["v"]))
    return t, fdir


def triplets(clip):
    """Frames with labels at f-1, f, f+1 and images on disk."""
    t, fdir = tracks(clip)
    out = []
    for f in t:
        if all(g in t for g in (f - 1, f + 1)) and all(
                os.path.exists(f"{fdir}/{g}.jpg") for g in (f - 1, f, f + 1)):
            out.append((clip, f))
    return out


def heatmap(u, v, sigma=SIGMA):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))


SN_MANIFEST = "data/soccernet_manifest.csv"
SN_SIGMA = 3.5   # SoccerNet ball labels carry 5-30 px jitter — soften targets


def sn_tracks():
    if not hasattr(sn_tracks, "_c"):
        c = {}
        for r in csv.DictReader(open(SN_MANIFEST)):
            c.setdefault(r["seq_dir"], {})[int(r["frame"])] = (
                float(r["u"]), float(r["v"]))
        sn_tracks._c = c
    return sn_tracks._c


def sn_samples(stride=2):
    """Manifest entries with labeled neighbors, as (("sn", seq_dir), f)."""
    out = []
    for seq, t in sn_tracks().items():
        fs = sorted(t)
        for f in fs[::stride]:
            if f - 1 in t and f + 1 in t:
                out.append((("sn", seq), f))
    return out


def _resolve(clip):
    """-> (track dict, frames dir, filename fmt, target sigma)."""
    if isinstance(clip, tuple) and clip[0] == "sn":
        return sn_tracks()[clip[1]], clip[1], "{:06d}.jpg", SN_SIGMA
    t, fdir = tracks._cache.setdefault(clip, tracks(clip))
    return t, fdir, "{}.jpg", SIGMA


def make_sample(clip, f, rng, negative=False):
    t, fdir, fmt, sigma = _resolve(clip)
    imgs = [cv2.imread(f"{fdir}/{fmt.format(g)}") for g in (f - 1, f, f + 1)]
    ih, iw = imgs[0].shape[:2]
    u, v = t[f]
    if negative:   # crop that excludes the ball
        for _ in range(20):
            x0 = rng.integers(0, max(1, iw - W))
            y0 = rng.integers(0, max(1, ih - H))
            if not (x0 - 20 < u < x0 + W + 20 and y0 - 20 < v < y0 + H + 20):
                break
    else:          # ball at a uniform random position inside the crop
        x0 = int(np.clip(u - rng.integers(20, W - 20), 0, iw - W))
        y0 = int(np.clip(v - rng.integers(20, H - 20), 0, ih - H))
    xs, ys_t = [], []
    flip = rng.random() < 0.5
    gain = rng.uniform(0.8, 1.2)
    bias = rng.uniform(-20, 20)
    for g, img in zip((f - 1, f, f + 1), imgs):
        c = img[y0:y0 + H, x0:x0 + W].astype(np.float32)
        c = np.clip(c * gain + bias, 0, 255)[:, :, ::-1] / 255.0
        gu, gv = t.get(g, (np.nan, np.nan))
        hm = np.zeros((H, W), np.float32)
        if not negative and np.isfinite(gu) and 0 <= gu - x0 < W and 0 <= gv - y0 < H:
            hm = heatmap(gu - x0, gv - y0, sigma).astype(np.float32)
        if flip:
            c, hm = c[:, ::-1], hm[:, ::-1]
        xs.append(((c - MEAN) / STD).transpose(2, 0, 1))
        ys_t.append(hm)
    return (np.concatenate(xs).astype(np.float32),
            np.stack(ys_t).astype(np.float32))


tracks._cache = {}


def batches(samples, rng, bs=8, neg_frac=0.25):
    idx = rng.permutation(len(samples))
    for i in range(0, len(idx) - bs + 1, bs):
        xs, ys = [], []
        for j in idx[i:i + bs]:
            clip, f = samples[j]
            x, y = make_sample(clip, f, rng, negative=rng.random() < neg_frac)
            xs.append(x), ys.append(y)
        yield (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)))


def wbce(logits, target, pos_w=150.0):
    w = 1.0 + (pos_w - 1.0) * target
    return (w * torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")).mean()


def evaluate(model, device, clips=None, stride=2):
    model.eval()
    print(f"{'clip':12s} {'n':>4} {'top1@5':>7} {'top5@5':>7} {'med@truth':>10}")
    for clip in (clips or CLIPS):
        t, fdir = tracks(clip)
        rows = sorted(t)
        hit1 = hitk = n = 0
        vals = []
        for f in rows[::stride]:
            if not all(os.path.exists(f"{fdir}/{g}.jpg") for g in (f-1, f, f+1)):
                continue
            u_t, v_t = t[f]
            img0 = cv2.imread(f"{fdir}/{f}.jpg")
            ih, iw = img0.shape[:2]
            x0 = int(min(max(0, u_t - 256), iw - W))
            y0 = int(min(max(0, v_t - 144), ih - H))
            xs = []
            for g in (f - 1, f, f + 1):
                c = cv2.imread(f"{fdir}/{g}.jpg")[y0:y0+H, x0:x0+W]
                c = c.astype(np.float32)[:, :, ::-1] / 255.0
                xs.append(((c - MEAN) / STD).transpose(2, 0, 1))
            x = torch.from_numpy(np.concatenate(xs)[None].astype(np.float32)).to(device)
            with torch.no_grad():
                hm = torch.sigmoid(model(x)[0][0, 1]).cpu()
            mp = torch.nn.functional.max_pool2d(hm[None, None], 5, 1, 2)[0, 0]
            pk = torch.nonzero((hm == mp) & (hm > 0.05), as_tuple=False)
            pv = hm[pk[:, 0], pk[:, 1]]
            order = torch.argsort(pv, descending=True)[:5]
            tu, tv = u_t - x0, v_t - y0
            ds = [np.hypot(pk[i][1].item() - tu, pk[i][0].item() - tv)
                  for i in order]
            if ds and ds[0] <= 5: hit1 += 1
            if any(d <= 5 for d in ds): hitk += 1
            vals.append(hm[int(tv), int(tu)].item())
            n += 1
        print(f"{clip:12s} {n:4d} {100*hit1/max(n,1):6.1f}% {100*hitk/max(n,1):6.1f}% "
              f"{np.median(vals):10.3f}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    rng = np.random.default_rng(0)
    if mode == "check":
        samp = [s for c in CLIPS if c != HELD_OUT for s in triplets(c)]
        print(f"train samples: {len(samp)} (held out: {HELD_OUT})")
        x, y = make_sample(*samp[0], rng)
        print("sample shapes:", x.shape, y.shape, "target max", y.max())
        vis = (y[1] * 255).astype(np.uint8)
        cv2.imwrite("reports/train_sample_heatmap.png",
                    np.hstack([cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR),
                               ((x[3:6].transpose(1, 2, 0) * STD + MEAN)
                                * 255)[:, :, ::-1].astype(np.uint8)]))
        print("wrote reports/train_sample_heatmap.png — LOOK at it")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_wasb().to(device)
    if mode == "eval":
        if len(sys.argv) > 2:
            ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
            model.load_state_dict(ck.get("model_state_dict", ck))
            model.to(device)
        evaluate(model, device)
        return
    # train: SoccerNet corpus + our verified labels oversampled 8x
    ours = [s for c in CLIPS if c != HELD_OUT for s in triplets(c)]
    sn = sn_samples(stride=2) if os.path.exists(SN_MANIFEST) else []
    samp = sn + ours * (8 if sn else 1)
    print(f"finetuning on {len(samp)} samples "
          f"({len(sn)} soccernet + {len(ours)}x8 ours; {HELD_OUT} held out), "
          f"{device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    step = 0
    n_epochs = 3 if sn else 12
    for epoch in range(n_epochs):
        losses = []
        for x, y in batches(samp, rng):
            x, y = x.to(device), y.to(device)
            out = model(x)[0].contiguous()
            loss = wbce(out, y.contiguous())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); step += 1
        print(f"epoch {epoch}: loss {np.mean(losses):.4f} ({step} steps)",
              flush=True)
    torch.save({"model_state_dict": model.state_dict()}, "models/wasb_ft.pth")
    print("saved models/wasb_ft.pth")
    evaluate(model, device)


if __name__ == "__main__":
    main()
