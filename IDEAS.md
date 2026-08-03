# Ideas — deferred, not in scope until explicitly pulled in

- Spin decay over the flight (currently omega is constant; decay is small
  over ~1 s but would matter for long diagonal deliveries).
- Anisotropic per-frame pixel noise from the tracker (motion blur stretches
  the ball along the velocity direction; the residual could weight axes
  differently).
- Rolling-shutter correction for broadcast cameras.
- Wind as a nuisance parameter with a tight prior.
- MCMC posterior (emcee) replacing the bootstrap on real data — planned in
  CLAUDE.md; the synthetic calibration factor then becomes a validation
  check instead of a correction.
- Keeper reachability envelope parameterized by reaction time and dive
  speed distributions from the literature, not single point values.
