# Free Kick Trajectory Reconstruction

Reconstructs 3D ball trajectories from monocular broadcast footage of free
kicks and compares technique across elite takers. The ball is tracked in 2D,
the camera is calibrated from pitch lines, and a drag + Magnus ODE is fit so
its projection matches the observed track — physics resolves the monocular
depth ambiguity.

## Honesty caveats

These hold for every number this project produces.

- **The sample is goals-only.** Compilations contain no misses, so the data
  can compare technique across players; it cannot say anything about what
  distinguishes makes from misses.
- **Spin is a posterior, never a point estimate.** Only the spin component
  transverse to the velocity (the part that produces Magnus force) is
  observable at all; the component along the flight direction is reported as
  unobservable. Transverse spin carries a known shrinkage bias (~15% low at
  2 px tracking noise) and its intervals are inflated by a
  synthetic-calibrated factor to reach nominal coverage.
- **Launch speed and direction are far better determined than spin** and
  lead every analysis.
- **The discard rate is reported.** Clips that fail tracking, calibration,
  or validation are counted, not silently dropped.

## Status

Phase 0 (synthetic validation) complete. The fitting model — z0 pinned at
ball radius, launch point derived from the first-frame ray, goal-plane
crossing as a measured hinge box in the residual, percentile bootstrap with
spin inflation — was selected by experiment; every choice is backed by a
rerunnable script:

```
python -m src.validate_recovery           # recovery test on the shipped model
python -m src.experiment_constraints      # why the endpoint constraint, not launch priors
python -m src.experiment_endpoint_stress  # box mis-centring stress + interval calibration
python -m src.experiment_noise_sweep      # widths / bias / calibration vs pixel noise
```

Key Phase 0 numbers (23 m curler, 25 fps, 2 px noise, crossing measured to
0.2 m): launch speed to ~±0.5 m/s, launch angles to ~±1°, transverse spin
interval ~±230 rpm around a ~15%-low estimate. A mis-centred crossing box is
the one configuration that produces confident wrong answers — box half-width
must never be tighter than the crossing measurement's real accuracy.

Next: Phase 1, ball tracking (SAM 2) on five real clips.
