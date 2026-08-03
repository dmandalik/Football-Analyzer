"""Flight model for a spinning football: drag + Magnus + gravity, RK4 integrator.

World frame: +z up, gravity along -z. Positions in metres, velocities in m/s,
spin as an angular velocity vector in rad/s (assumed constant over the flight;
spin decay over ~1 s of flight is small compared to our other uncertainties).
"""

import numpy as np

# FIFA size 5 ball
MASS = 0.43                      # kg
DIAMETER = 0.22                  # m
RADIUS = DIAMETER / 2.0
AREA = np.pi * RADIUS ** 2       # ~0.038 m^2 cross-section
RHO = 1.225                      # kg/m^3 air density
GRAVITY = np.array([0.0, 0.0, -9.81])

# Drag crisis: C_d falls from ~0.43 (subcritical) to ~0.15 (supercritical)
# around Re ~ 2.2e5, which for a 0.22 m ball is ~15 m/s. Modelled as a
# logistic transition in speed so the force is smooth for the optimizer.
CD_SUBCRITICAL = 0.43
CD_SUPERCRITICAL = 0.15
DRAG_CRISIS_SPEED = 15.0         # m/s, centre of the transition
DRAG_CRISIS_WIDTH = 2.5          # m/s, transition width


def drag_coefficient(speed):
    """Velocity-dependent C_d through the drag crisis."""
    return CD_SUPERCRITICAL + (CD_SUBCRITICAL - CD_SUPERCRITICAL) / (
        1.0 + np.exp((speed - DRAG_CRISIS_SPEED) / DRAG_CRISIS_WIDTH)
    )


def lift_coefficient(spin_factor):
    """C_l as a saturating function of spin factor S = R*omega/v.

    Tuned so C_l ~ 0.13 at S=0.1 and ~0.25 at S=0.3, matching the 0.1-0.3
    range reported for footballs at typical free-kick spin factors.
    """
    return 0.45 * spin_factor / (0.25 + spin_factor)


def acceleration(velocity, omega):
    """Net acceleration on the ball: drag + Magnus + gravity."""
    speed = np.linalg.norm(velocity)
    if speed < 1e-9:
        return GRAVITY.copy()

    dyn = 0.5 * RHO * AREA * speed ** 2 / MASS  # dynamic pressure term / mass
    accel = GRAVITY - drag_coefficient(speed) * dyn * velocity / speed

    omega_mag = np.linalg.norm(omega)
    if omega_mag > 1e-9:
        cross = np.cross(omega, velocity)
        cross_mag = np.linalg.norm(cross)
        if cross_mag > 1e-12:
            spin_factor = RADIUS * omega_mag / speed
            accel = accel + lift_coefficient(spin_factor) * dyn * cross / cross_mag
    return accel


def _rk4_step(pos, vel, omega, dt):
    k1v = acceleration(vel, omega)
    k1p = vel
    k2v = acceleration(vel + 0.5 * dt * k1v, omega)
    k2p = vel + 0.5 * dt * k1v
    k3v = acceleration(vel + 0.5 * dt * k2v, omega)
    k3p = vel + 0.5 * dt * k2v
    k4v = acceleration(vel + dt * k3v, omega)
    k4p = vel + dt * k3v
    new_pos = pos + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
    new_vel = vel + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return new_pos, new_vel


def simulate(p0, v0, omega, times, max_dt=1.0 / 60.0):
    """Integrate the flight and return positions at the requested times.

    times must be increasing, with times[0] corresponding to state (p0, v0).
    Each inter-sample interval is split into RK4 substeps no longer than
    max_dt, landing exactly on every sample time (deterministic output).
    Default max_dt: RK4 at 1/60 s matches a 1/960 s reference to ~1e-10 m
    over a full free-kick flight, far below any measurement noise.
    """
    times = np.asarray(times, dtype=float)
    pos = np.array(p0, dtype=float)
    vel = np.array(v0, dtype=float)
    omega = np.array(omega, dtype=float)

    out = np.empty((len(times), 3))
    out[0] = pos
    for i in range(1, len(times)):
        span = times[i] - times[i - 1]
        n_sub = max(1, int(np.ceil(span / max_dt)))
        dt = span / n_sub
        for _ in range(n_sub):
            pos, vel = _rk4_step(pos, vel, omega, dt)
        out[i] = pos
    return out


def time_to_plane_y0(p0, v0, omega, t_max=3.0, dt=1.0 / 240.0):
    """Flight time until the ball crosses the goal plane y=0 (moving in -y).

    Returns None if the plane is not crossed within t_max. Used to pick how
    many frames of a synthetic clip to generate.
    """
    pos = np.array(p0, dtype=float)
    vel = np.array(v0, dtype=float)
    omega = np.array(omega, dtype=float)
    t = 0.0
    while t < t_max:
        prev_y = pos[1]
        pos, vel = _rk4_step(pos, vel, omega, dt)
        t += dt
        if prev_y > 0.0 >= pos[1]:
            # linear interpolation inside the last step
            frac = prev_y / (prev_y - pos[1])
            return t - dt + frac * dt
    return None
