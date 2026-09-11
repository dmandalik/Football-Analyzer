# Kaggle training runbook

Why: 20-epoch full training = ~2.5h on a T4 vs ~20h wall on a sleeping
MacBook. Your ~30 GPU-h/week covers a dozen iterations.

Steps: see the docstring of train_kaggle.py. Key policy points:
- The SoccerNet NDA password lives ONLY in Kaggle Secrets (never in code,
  never in this repo, never in a dataset).
- The NDA'd videos are downloaded by the notebook onto the ephemeral VM
  and die with it. Never attach them as a Kaggle dataset, private or not.
- The "freekick-labels" dataset you create contains only OUR OWN frames
  and labels (your work product) - safe to store privately.

Local drop-in after a run:
  cp ~/Downloads/wasb_ft.pth models/wasb_ft.pth
  python -m src.train_ball_detector eval   # fixed benchmark
