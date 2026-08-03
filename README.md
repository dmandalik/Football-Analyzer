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
  unobservable. Transverse spin carries a noise-dependent shrinkage bias
  (from ~0 at 0.5 px to ~-290 rpm at 5 px tracking noise); its interval is
  recentred and inflated by a correction keyed to each clip's own measured
  reprojection RMS, verified on held-out synthetic data to hold ~68%
  coverage across the 0.5-5 px range.
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
python -m src.experiment_spin_calibration calibrate|verify  # rms-keyed spin correction
python -m src.experiment_geometry_sweep   # precision vs camera viewing angle
```

Key Phase 0 numbers (23 m curler, 25 fps, 2 px noise, crossing measured to
0.2 m): launch speed to ~±0.5 m/s, launch angles to ~±1°, transverse spin
interval ~±190 rpm after the rms-keyed debias. A mis-centred crossing box is
the one configuration that produces confident wrong answers — box half-width
must never be tighter than the crossing measurement's real accuracy.

Camera geometry (viewing angle alpha between camera axis and flight
direction) sets what a clip can support: side-on views (alpha ~90) give the
best speed (±0.1 m/s) but the worst azimuth and spin (curl lies along the
viewing ray); views from behind the kicker (alpha ~10) invert that; the
typical broadcast diagonal (alpha 25-40) is the balanced optimum and the
only geometry where spin claims are near their best. Clip grading in
Phase 1 records the measured viewing angle per clip.

Next: Phase 1, ball tracking (SAM 2) on five real clips.
