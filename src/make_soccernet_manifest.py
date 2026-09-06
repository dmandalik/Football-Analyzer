"""Extract ball-center labels from SoccerNet-Tracking into one manifest.

Reads each SNMOT sequence's gameinfo.ini (which tracklet is the ball)
and gt.txt (MOT boxes), writes data/soccernet_manifest.csv with rows
seq_dir,frame,u,v. Ball boxes are known noisy: boxes larger than
MAX_BOX px in either dimension are dropped (players mislabeled as ball),
and sequences whose "ball" is >15% oversized are reported.

NDA note: labels only — no SoccerNet imagery leaves data/soccernet/.

Run from repo root:  python -m src.make_soccernet_manifest
"""

import configparser
import csv
import glob
import os

MAX_BOX = 60.0
ROOT = "data/soccernet/tracking/train"


def main():
    rows, report = [], []
    for seq in sorted(glob.glob(f"{ROOT}/SNMOT-*")):
        gi = configparser.ConfigParser()
        gi.read(f"{seq}/gameinfo.ini")
        sec = gi["Sequence"] if "Sequence" in gi else gi[gi.sections()[0]]
        ball_ids = {k.split("_")[1] for k, v in sec.items()
                    if k.startswith("trackletid") and
                    v.split(";")[0].strip().lower() == "ball"}
        if not ball_ids:
            report.append((os.path.basename(seq), "NO BALL TRACKLET", 0))
            continue
        n_ok = n_big = 0
        for line in open(f"{seq}/gt/gt.txt"):
            p = line.strip().split(",")
            if p[1] in ball_ids:
                x, y, w, h = map(float, p[2:6])
                if w > MAX_BOX or h > MAX_BOX:
                    n_big += 1
                    continue
                rows.append((f"{seq}/img1", int(p[0]),
                             round(x + w / 2, 1), round(y + h / 2, 1)))
                n_ok += 1
        report.append((os.path.basename(seq), f"{n_ok} boxes", n_big))
    os.makedirs("data", exist_ok=True)
    with open("data/soccernet_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_dir", "frame", "u", "v"])
        w.writerows(rows)
    big_total = sum(r[2] for r in report)
    print(f"{len(rows)} ball labels from {len(report)} sequences "
          f"({big_total} oversized boxes dropped)")
    for name, msg, nb in report:
        if "NO BALL" in msg or nb > 100:
            print(f"  WARN {name}: {msg}, {nb} oversized")


if __name__ == "__main__":
    main()
