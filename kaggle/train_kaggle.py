"""Full detector training on Kaggle (T4/P100), self-contained.

Setup (one-time, ~10 min):
  1. Kaggle notebook: New Notebook -> Settings: GPU T4, Internet ON.
  2. Add-ons > Secrets: add SOCCERNET_PW = your NDA password.
  3. Create a small private Kaggle dataset "freekick-labels" containing:
       soccernet_manifest.csv  (regenerated here anyway - optional)
       confusers.csv           from data/confusers.csv
       our_tracks/             the 5 verified track CSVs
       our_frames/             ONLY the labeled frames referenced by the
                               tracks (run scripts/pack_labels.py to build
                               this folder, ~200MB)
     Attach it to the notebook as input.
  4. Paste this file into a cell, run. ~2.5h for 20 epochs on T4.
     Output: /kaggle/working/wasb_ft.pth (+ per-epoch checkpoints).
     Download and drop into models/wasb_ft.pth locally.

The SoccerNet download (~17GB tracking train+test) happens on Kaggle's
side using YOUR password from the secret store - the NDA'd data never
transits anywhere except SoccerNet -> Kaggle VM, and is deleted with the
VM. Do not attach it as a public dataset.
"""
import os, subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "SoccerNet", "omegaconf", "gdown"], check=True)
from kaggle_secrets import UserSecretsClient
pw = UserSecretsClient().get_secret("SOCCERNET_PW")

from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory="/kaggle/tmp/soccernet")
dl.password = pw
dl.downloadDataTask(task="tracking", split=["train", "test"])
for z in ("train", "test"):
    subprocess.run(["unzip", "-q", "-o",
                    f"/kaggle/tmp/soccernet/tracking/{z}.zip",
                    "-d", "/kaggle/tmp/soccernet/tracking/"], check=True)

subprocess.run(["git", "clone", "-q", "--depth", "1",
                "https://github.com/nttcom/WASB-SBDT",
                "/kaggle/tmp/WASB-SBDT"], check=True)
subprocess.run(["gdown", "-q", "1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l",
                "-O", "/kaggle/tmp/wasb_soccer_best.pth.tar"], check=True)

# --- from here: the repo's trainer, paths adapted via env ---
os.environ["FREEKICK_KAGGLE"] = "1"
print("environment ready - now run the training cell "
      "(copy src/train_ball_detector.py with KAGGLE path overrides)")
