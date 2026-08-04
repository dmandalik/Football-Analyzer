# Free Kick Trajectory Reconstruction — Project Brief

## What this project is

Reconstruct 3D ball trajectories from monocular broadcast footage of free kicks, and compare
technique across elite takers (Messi, Ronaldo, Juninho, Pirlo, Ward-Prowse, Çalhanoğlu, Şahin).

Method: detect/track the ball in 2D, calibrate the camera from pitch lines, then fit a
drag + Magnus ODE whose projection matches the observed 2D track. Physics resolves the
monocular depth ambiguity — we are not guessing depth, we are constraining it.

Output: an interactive 3D visualization of trajectory bundles with keeper reachability envelopes.

## Working principles — read these every session

1. **Every session ends with something runnable or visible.** A plot, a rendered overlay, a
   number with an error bar. Never end a session with "the refactor is in progress."
2. **Synthetic before real.** Any component gets validated on data where we know the true
   answer before it touches footage.
3. **Small loops.** One component, one validation, then stop and look at the output.
   Long autonomous runs on this codebase drift and the failures are silent.
4. **Failures here don't throw exceptions.** Bad homography produces players in the stands.
   Bad ball tracking produces a smooth path across grass. Always render and look.
5. **Don't expand scope.** If an idea comes up mid-session, write it to IDEAS.md and continue.

## Non-negotiable honesty requirements

These must survive into the final output. Do not let them get dropped during refactors.

- **Spin is reported as a posterior with a credible interval, never a point estimate.**
  Trajectory-only spin estimation is considered ill-posed in the literature — Magnus curvature
  is confounded with ball-position measurement noise. Reliable spin needs bounce behavior,
  visible ball-surface pattern, or event cameras. We are producing a best-fit under uncertainty.
- **Sample is goals-only.** Compilations do not contain misses. We can compare technique
  *across players*; we cannot say what distinguishes makes from misses. This caveat appears
  in the README and on the visualization.
- **Launch velocity and angle are far better determined than spin.** Lead with those.
- **Report the discard rate.** How many clips were collected, how many survived tracking,
  calibration, and goal-mouth validation. A project that reports 28/60 usable is credible.
  One that silently reports 28 is not.

## Physics

```
m·v̇ = -k_D·‖v‖·v  +  k_M·(ω × v)  +  m·g
        drag           Magnus         gravity
```

Constants (FIFA size 5):
- diameter ≈ 0.22 m, cross-sectional area A ≈ 0.038 m², mass ≈ 0.43 kg
- Drag: F_D = ½·C_d·A·ρ·v², air density ρ ≈ 1.225 kg/m³
- **C_d is velocity-dependent.** ~0.43 subcritical, dropping to ~0.15 supercritical through
  the drag crisis at Reynolds ≈ 2.2×10⁵ (roughly 15 m/s). A fixed C_d is wrong and will
  bias every fit. Implement the transition.
- Lift/Magnus coefficient C_l ≈ 0.1–0.3, depending on spin factor S = Rω/v

Integrate with RK4. Fit with `scipy.optimize.least_squares` minimizing reprojection error
between the projected 3D path and the observed 2D track.

Free parameters: initial position (3), launch velocity vector (3), spin vector (3).

## Validation — free ground truth

Every clip in the dataset is a goal. The reconstructed trajectory must pass through the goal
mouth. Discard any fit whose trajectory misses by more than ~0.5 m. This is a free, strong
check that costs nothing and catches most solver failures.

Secondary check: project a known pitch distance (penalty spot to goal line, 11 m) through the
homography and verify it comes out correct. Calibration that runs without error can still be wrong.

## Stack

- Ball tracking: SAM 2 (click-to-propagate) first; TrackNetV3 if motion blur defeats it
- Calibration: PnLCalib (integrated in SoccerNet's sn-gamestate repo — use their working code,
  don't reimplement from the paper)
- Physics: numpy + scipy
- Clips: yt-dlp
- Frontend: three.js reading precomputed static JSON

## Architecture constraint

**Precompute everything offline. Ship static JSON. The UI does zero inference.**
The frontend must be hostable on a free static tier forever. If the demo needs a GPU at
request time, it is wrong.

## Compute

Kaggle free tier (~30 GPU-hrs/week, T4/P100, 9-hour sessions with background execution).
Total project need is roughly 5 GPU-hours — this is not a compute-constrained project.
Still: checkpoint per-clip so a timeout never costs a full run.

## Repo layout

```
data/
  raw_clips/          # yt-dlp output, gitignored
  clip_manifest.csv   # player, source, grade (usable/partial/unusable), notes
  tracks/             # 2D ball tracks per clip
  calibrations/       # homography per clip
  fits/               # ODE fit results + posteriors
src/
  physics.py          # RK4 integrator, force model
  fitting.py          # reprojection residual, least_squares wrapper
  synthetic.py        # ground-truth trajectory generator for validation
  tracking.py         # SAM2 wrapper
  calibration.py      # PnLCalib wrapper
web/                  # three.js frontend
IDEAS.md              # scope creep goes here, not into the code
```

## Build order — do not reorder

**Phase 0 — synthetic validation (no footage needed).**
Build physics.py, fitting.py, synthetic.py. Generate a trajectory with known launch velocity
and spin, project it to 2D, add pixel noise, and verify the solver recovers the true parameters.
Then sweep noise level and plot how the spin posterior widens.

This tells us how much spin uncertainty is inherent to the method *before* collecting any data.
It is the most valuable thing in the project and it requires nothing external.

**Gate: if spin is unrecoverable even at low synthetic noise, reframe the project around
launch velocity and trajectory shape only.**

**Phase 1 — tracking gate.** Five real clips. Does SAM 2 follow the ball through motion blur?
Render every track as an overlay video and watch it.

**Gate: if tracking fails on compilation footage, try TrackNetV3. If that also fails, stop
and reconsider.**

**Phase 2 — calibration.** PnLCalib on the same five clips. Verify with the 11 m check.

**Phase 3 — end-to-end on five clips.** Including goal-mouth validation.

**Phase 4 — scale to the full set.** Batch, discard failures, characterize uncertainty.

**Phase 5 — frontend.**

## Research questions

Ordered by how confidently they can be answered.

1. **How do launch speed and angle distributions differ across elite takers?**
   Best-determined quantities. Should produce a clear, defensible result.

2. **Do free-kick trajectories cluster into distinct shape families?**
   Late dip vs. sustained curve vs. flat-and-fast. Cluster on trajectory descriptors.
   Are the clusters player-specific or shared?

3. **Where in the goal does each taker aim, and how much of the goal was physically reachable?**
   Compute keeper reachability from diving speed and reaction time given each flight path.
   This is the headline visual.

4. **How much curvature does each taker generate, and can we distinguish them given our
   uncertainty?** The honest version of the spin question. The answer may be "no" for some
   pairs — that is a legitimate finding.

5. **How much spin uncertainty is inherent to monocular trajectory fitting at broadcast
   frame rate?** Answered entirely by Phase 0. Publishable in its own right and independent
   of whether the football analysis works.

Explicitly out of scope: what distinguishes makes from misses (no miss data), claims that one
player's technique is superior, precise spin rates.
UPDATED: 
- Spin is reported as ω⊥ with interval plus ω∥ flagged unobservable; launch position enters as a measured prior, not a free parameter; uncertainty via MCMC on real data.
- Goal-plane crossing is a measurement in the residual, not a discard filter. z0 fixed at ball radius; x0/y0 follow from the first-frame ray.
- Spin reported as ω⊥ with interval and ω∥ as unobservable, with the measured widths quoted. The honesty requirement now has numbers behind it.
- Clip grading keys on CROSS-TRACK residual RMS, not total RMS. Real tracking noise is anisotropic (blur smears the centroid along the travel direction 3–11×; curvature information lives cross-track). Threshold: cross-track ≤ 2 px supports spin claims — all three Phase 1 clips pass (1.2–1.8 px) even though their total RMS is 5–7 px. The spin bias/inflation correction is keyed on measured cross-track RMS (table in fitting.py). The earlier ratio-dependence of the bias was a scalar-whitening artifact — resolved by anisotropic whitening (fit_flight default); single-key tables are valid, re-derive them under whitening or clip-match per fit-grade clip.