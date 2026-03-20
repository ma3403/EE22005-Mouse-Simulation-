"""
Author: ma3403
Date:   13th March 2026
Mouse Race Lap Time Simulator
==============================
EE22005 Engineering Practice and Design - Project Week 3

This simulation models the lap time of an autonomous differential-drive
'mouse' robot completing a 20m race track. The model is built progressively
in three stages:

    Stage 1 - Kinematic Model:
        Models the track as discrete segments. The mouse travels at a
        constant cruise speed, with reduced speed on bends and ramps.

    Stage 2 - Motor & Acceleration Model:
        Adds DC motor torque/speed curves and models acceleration from
        standstill and the effect of gradient load on motor speed.

    Stage 3 - PID Differential Steering Model:
        Adds a PID controller acting on the differential between left
        and right motor speeds to model realistic cornering behaviour
        and its effect on lap time.

"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS & PHYSICAL PARAMETERS
# (Values updated with lab measurements)
# ---------------------------------------------------------------------------

# --- Mouse Physical Parameters ---
MOUSE_MASS_KG       = 0.4        # kg  (update from lab measurement; 4kg is unrealistic)
WHEEL_DIAMETER_M    = 0.040      # m   (40 mm as specified)
WHEEL_RADIUS_M      = WHEEL_DIAMETER_M / 2
WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER_M  # m per revolution

# --- Motor Parameters (at 6 V, estimated from small hobby DC motors) ---
MOTOR_VOLTAGE_V         = 6.0    # V
MOTOR_NO_LOAD_RPM       = 780    # RPM  (measure with tachometer in lab)
MOTOR_STALL_TORQUE_NM   = 0.03   # N·m  (estimate; update from datasheet)
MOTOR_NO_LOAD_SPEED_RPS = MOTOR_NO_LOAD_RPM / 60  # rev/s

# Derived: no-load wheel surface speed (m/s)
MOTOR_MAX_SPEED_MS = MOTOR_NO_LOAD_SPEED_RPS * WHEEL_CIRCUMFERENCE

# --- Track Geometry ---
RAMP_ANGLE_DEG  = 15.0
RAMP_ANGLE_RAD  = math.radians(RAMP_ANGLE_DEG)
GRAVITY_MS2     = 9.81

# --- Lateral Friction (Cornering Limit) ---
# Coefficient of lateral (side) friction between rubber wheels and wood/MDF surface.
# This determines the maximum speed the mouse can travel around a curve before
# centripetal force exceeds the available friction and the mouse slides off track.
# Typical value for rubber on wood: μ_lateral ≈ 0.5
# Source: Engineering Toolbox — friction coefficients for rubber on wood.
MU_LATERAL = 0.5


def max_cornering_speed(radius: float) -> float:
    """
    Calculate the maximum safe cornering speed for a given turning radius.

    Derived from the condition that centripetal acceleration must not exceed
    the maximum lateral friction force the wheels can provide:

        m * v² / r  ≤  μ * m * g
        =>  v_max  =  √(μ * g * r)

    Note: mass cancels, so this limit is independent of mouse weight.

    Parameters
    ----------
    radius : float
        Turning radius of the curve in metres.

    Returns
    -------
    float
        Maximum safe cornering speed in m/s.
    """
    return math.sqrt(MU_LATERAL * GRAVITY_MS2 * radius)

# --- PID Controller Gains (Stage 3) ---
# Tune these to match the behaviour of your Arduino PID implementation
KP = 1.2   # Proportional gain
KI = 0.01  # Integral gain
KD = 0.05  # Derivative gain

# --- Simulation Time Step ---
DT = 0.001  # seconds (1 ms resolution)


# ===========================================================================
# STAGE 1 — TRACK SEGMENT CLASSES (Kinematic Model)
# ===========================================================================

class TrackSegment:
    """
    Base class representing a single segment of the race track.

    Each segment has a length and a speed factor which scales the mouse's
    cruise speed while traversing that segment. Subclasses override
    speed_factor() to implement segment-specific physics.

    Attributes
    ----------
    length : float
        Length of the segment in metres.
    name : str
        Human-readable label for the segment (used in output).
    """

    def __init__(self, length: float, name: str):
        """
        Initialise a TrackSegment.

        Parameters
        ----------
        length : float
            Length of the segment in metres.
        name : str
            Descriptive name for the segment.
        """
        self.length = length
        self.name   = name

    def speed_factor(self) -> float:
        """
        Return a multiplier (0.0–1.0) applied to the cruise speed.

        Straight segments return 1.0 (full speed). Subclasses reduce this
        to model the effect of corners, ramps, and slaloms.

        Returns
        -------
        float
            Speed multiplier for this segment type.
        """
        return 1.0  # Default: full cruise speed on a straight

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', length={self.length:.2f}m)"


class StraightSegment(TrackSegment):
    """A flat, straight section of track. Mouse travels at full cruise speed."""

    def speed_factor(self) -> float:
        """Full speed on a straight."""
        return 1.0


class BendSegment(TrackSegment):
    """
    A curved section of track (e.g. semicircles at each end).

    The maximum speed through a bend is physically limited by the lateral
    friction between the wheels and the track surface. If the mouse travels
    faster than this limit, centripetal force exceeds the available friction
    and the mouse will slide off the track.

    The cornering speed limit is derived from:
        v_max = sqrt(mu_lateral * g * radius)

    where mu_lateral = 0.5 (rubber wheels on wood/MDF track surface).

    Attributes
    ----------
    radius : float
        Turning radius of the bend in metres.
    v_max_corner : float
        Physics-derived maximum safe cornering speed in m/s.
    """

    def __init__(self, length: float, name: str, radius: float):
        """
        Initialise a BendSegment.

        Parameters
        ----------
        length : float
            Arc length of the bend in metres.
        name : str
            Descriptive name.
        radius : float
            Turning radius in metres.
        """
        super().__init__(length, name)
        self.radius       = radius
        # Pre-compute the physics-based cornering speed limit for this radius
        self.v_max_corner = max_cornering_speed(radius)

    def speed_factor(self) -> float:
        """
        Return a speed factor based on the centripetal friction limit.

        Rather than an arbitrary multiplier, this computes the ratio of the
        physics-derived cornering limit to the motor's top speed. The result
        is clamped to [0, 1] so it always acts as a reduction or neutral factor.

        Note: Stages 2 and 3 use v_max_corner directly for hard speed capping
        with a printed warning if the motor would otherwise exceed it.

        Returns
        -------
        float
            Speed multiplier for Stage 1 kinematic model.
        """
        # Express the cornering limit as a fraction of motor top speed
        factor = self.v_max_corner / MOTOR_MAX_SPEED_MS
        return min(factor, 1.0)  # Cannot exceed 1.0 (no speed-up on a bend)


class RampSegment(TrackSegment):
    """
    A section of track with a fixed gradient (uphill or downhill).

    Stage 1: Speed is scaled by a fixed factor based on gradient direction.
    Stage 2 replaces this with physics-based motor load calculations.

    Attributes
    ----------
    angle_deg : float
        Gradient angle in degrees (positive = uphill, negative = downhill).
    """

    def __init__(self, length: float, name: str, angle_deg: float):
        """
        Initialise a RampSegment.

        Parameters
        ----------
        length : float
            Length of the ramp along its slope in metres.
        name : str
            Descriptive name.
        angle_deg : float
            Ramp angle in degrees. Positive = uphill, negative = downhill.
        """
        super().__init__(length, name)
        self.angle_deg = angle_deg
        self.angle_rad = math.radians(angle_deg)

    def speed_factor(self) -> float:
        """
        Scale speed based on ramp direction.

        Downhill: slight speed increase (1.1×).
        Uphill:   significant speed reduction due to gravity load (0.6×).
        Stage 2 replaces these heuristics with torque-based calculations.
        """
        if self.angle_deg < 0:
            return 1.10  # Downhill — gravity assists
        else:
            return 0.60  # Uphill — gravity opposes, motor works harder


class SlalomSegment(TrackSegment):
    """
    A sinusoidal S-bend section of track.

    Like BendSegment, the maximum safe speed is governed by the centripetal
    friction limit. The slalom uses a tighter radius (0.5m) than the
    semicircular ends, giving a lower cornering speed limit.

    Additionally, because the mouse must continuously alternate steering
    direction through each S-bend, an extra 15% speed reduction is applied
    on top of the friction limit to model the transient steering losses
    during direction reversal.

    Attributes
    ----------
    bend_radius : float
        Approximate radius of curvature of each S-bend in metres.
    num_bends : int
        Number of complete S-bends in the section.
    v_max_corner : float
        Physics-derived maximum safe cornering speed in m/s.
    """

    def __init__(self, length: float, name: str, bend_radius: float, num_bends: int):
        """
        Initialise a SlalomSegment.

        Parameters
        ----------
        length : float
            Total path length of the slalom section in metres.
        name : str
            Descriptive name.
        bend_radius : float
            Radius of curvature for each S-bend in metres.
        num_bends : int
            Number of complete S-bends.
        """
        super().__init__(length, name)
        self.bend_radius  = bend_radius
        self.num_bends    = num_bends
        # Cornering limit based on the tightest radius in the slalom
        self.v_max_corner = max_cornering_speed(bend_radius)

    def speed_factor(self) -> float:
        """
        Return a speed factor combining the centripetal friction limit and
        a steering-reversal penalty.

        The base limit is v_max_corner / motor_top_speed (same as BendSegment).
        An additional 15% reduction accounts for the transient loss of forward
        speed each time the mouse reverses steering direction at an S-bend apex.

        Returns
        -------
        float
            Combined speed multiplier for Stage 1 kinematic model.
        """
        base_factor    = self.v_max_corner / MOTOR_MAX_SPEED_MS
        steering_penalty = 0.85  # 15% reduction for direction-reversal transients
        return min(base_factor * steering_penalty, 1.0)


# ===========================================================================
# TRACK DEFINITION
# Full 20m race track broken into segments (clockwise from Start)
# ===========================================================================

def build_track() -> list:
    """
    Construct the list of TrackSegments representing the full race circuit.

    Segment order (clockwise from Start):
        1. Right semicircle (start bend)
        2. Downhill ramp (15°)
        3. Slalom section (2 S-bends, 0.5m radius)
        4. Left semicircle (far end bend)
        5. Short straight
        6. Uphill ramp (15°)
        7. Final curve back to start

    Returns
    -------
    list of TrackSegment
        Ordered list of all track segments.
    """
    segments = [
        BendSegment    (length=1.88, name="Start semicircle (right bend)",  radius=0.6),
        RampSegment    (length=1.50, name="Downhill ramp 15°",              angle_deg=-15.0),
        SlalomSegment  (length=5.50, name="Slalom section (2 S-bends)",     bend_radius=0.5, num_bends=2),
        BendSegment    (length=1.88, name="Far-end semicircle (left bend)", radius=0.6),
        StraightSegment(length=1.74, name="Short straight"),
        RampSegment    (length=1.50, name="Uphill ramp 15°",               angle_deg=+15.0),
        BendSegment    (length=1.00, name="Final curve to finish",          radius=0.6),
    ]
    # Sanity check: total track length should be ~20m
    total = sum(s.length for s in segments)
    print(f"[Track] Total track length: {total:.2f} m")
    print(f"[Physics] Cornering speed limits (μ={MU_LATERAL}, wood/MDF surface):")
    for s in segments:
        if hasattr(s, 'v_max_corner'):
            print(f"  {s.name:<40} v_max = {s.v_max_corner:.3f} m/s")
    return segments


# ===========================================================================
# STAGE 1 — KINEMATIC LAP TIME ESTIMATOR
# ===========================================================================

class KinematicModel:
    """
    Stage 1: Constant-speed kinematic model of lap time.

    The mouse is assumed to travel at a fixed cruise speed, scaled per
    segment by that segment's speed_factor(). No acceleration or motor
    physics are considered — this gives a best-case lower bound on lap time.

    Attributes
    ----------
    cruise_speed : float
        Maximum travel speed of the mouse in m/s.
    track : list of TrackSegment
        Ordered list of track segments.
    """

    def __init__(self, cruise_speed: float, track: list):
        """
        Initialise the kinematic model.

        Parameters
        ----------
        cruise_speed : float
            Cruise speed in m/s. Cannot exceed MOTOR_MAX_SPEED_MS.
        track : list of TrackSegment
            The track to simulate.
        """
        self.cruise_speed = cruise_speed
        self.track        = track

    def simulate(self, verbose: bool = True) -> float:
        """
        Run the Stage 1 kinematic simulation and return the predicted lap time.

        For each segment, time = length / (cruise_speed × speed_factor).

        Parameters
        ----------
        verbose : bool
            If True, print a breakdown of time spent in each segment.

        Returns
        -------
        float
            Total predicted lap time in seconds.
        """
        print("\n" + "="*60)
        print("STAGE 1 — KINEMATIC MODEL")
        print(f"Cruise speed: {self.cruise_speed:.2f} m/s")
        print("="*60)

        total_time = 0.0

        for seg in self.track:
            # Effective speed through this segment
            effective_speed = self.cruise_speed * seg.speed_factor()

            # Avoid division by zero if speed_factor is 0
            if effective_speed <= 0:
                raise ValueError(f"Effective speed is zero for segment: {seg.name}")

            seg_time = seg.length / effective_speed
            total_time += seg_time

            if verbose:
                print(f"  {seg.name:<40} | "
                      f"{seg.length:.2f}m | "
                      f"speed={effective_speed:.2f}m/s | "
                      f"time={seg_time:.3f}s")

        print(f"\n  >>> Stage 1 Predicted Lap Time: {total_time:.3f} seconds\n")
        return total_time


# ===========================================================================
# STAGE 2 — MOTOR & ACCELERATION MODEL
# ===========================================================================

class MotorModel:
    """
    Models a DC motor using a linear torque-speed characteristic.

    For a permanent magnet DC motor, torque decreases linearly with speed:
        torque(speed) = stall_torque × (1 - speed / no_load_speed)

    This is used to compute the motor's available torque at any given wheel
    speed, and to calculate the acceleration of the mouse.

    Attributes
    ----------
    stall_torque : float
        Maximum torque at zero speed (N·m).
    no_load_speed : float
        Maximum speed under no load (m/s at wheel surface).
    mass : float
        Mass of the mouse in kg (used for F = ma calculations).
    """

    def __init__(self, stall_torque: float, no_load_speed: float, mass: float):
        """
        Initialise the motor model.

        Parameters
        ----------
        stall_torque : float
            Stall torque in N·m.
        no_load_speed : float
            No-load wheel surface speed in m/s.
        mass : float
            Mouse mass in kg.
        """
        self.stall_torque  = stall_torque
        self.no_load_speed = no_load_speed
        self.mass          = mass

    def torque_at_speed(self, speed: float) -> float:
        """
        Compute motor torque at a given wheel surface speed.

        Uses the linear DC motor characteristic:
            T(v) = T_stall × (1 - v / v_no_load)

        Parameters
        ----------
        speed : float
            Current wheel surface speed in m/s.

        Returns
        -------
        float
            Motor torque in N·m. Clamped to zero if speed exceeds no-load speed.
        """
        # Clamp: motor cannot pull if already at or above no-load speed
        speed = max(0.0, min(speed, self.no_load_speed))
        return self.stall_torque * (1.0 - speed / self.no_load_speed)

    def drive_force(self, speed: float) -> float:
        """
        Compute the net drive force at the wheel contact patch.

        Force = Torque / Wheel_radius (both motors combined).

        Parameters
        ----------
        speed : float
            Current speed in m/s.

        Returns
        -------
        float
            Net drive force in Newtons (both motors combined).
        """
        # Two motors driving two wheels
        torque_per_motor = self.torque_at_speed(speed)
        force_per_wheel  = torque_per_motor / WHEEL_RADIUS_M
        return 2.0 * force_per_wheel  # Two driven wheels

    def gravity_force(self, angle_rad: float) -> float:
        """
        Compute the component of gravity opposing motion on a ramp.

        F_gravity = m × g × sin(θ)
        Positive angle = uphill (opposing force); negative = downhill (assisting).

        Parameters
        ----------
        angle_rad : float
            Ramp angle in radians. Positive = uphill.

        Returns
        -------
        float
            Gravity force component along the slope in Newtons.
            Positive value opposes forward motion (uphill).
        """
        return self.mass * GRAVITY_MS2 * math.sin(angle_rad)

    def simulate_segment(self, segment: TrackSegment) -> tuple:
        """
        Simulate the mouse traversing one segment using Newton's 2nd law.

        Uses Euler integration with time step DT to propagate speed and
        position. At each time step:
            a = (F_drive - F_gravity - F_friction) / mass

        For curved segments (BendSegment, SlalomSegment), the speed is
        hard-capped at v_max_corner derived from the centripetal friction
        limit:
            v_max = sqrt(mu_lateral * g * radius)

        If the motor would otherwise push the mouse beyond this limit, a
        warning is printed and speed is clamped — representing the real-world
        consequence of the mouse sliding off the track at that corner.

        Parameters
        ----------
        segment : TrackSegment
            The segment to simulate.

        Returns
        -------
        tuple of (float, float)
            (time_taken, exit_speed) — time in seconds, exit speed in m/s.
        """
        # Rolling friction coefficient (estimated; update from lab)
        MU_ROLL = 0.05

        # Get gradient angle — only RampSegments have an angle attribute
        angle_rad = getattr(segment, 'angle_rad', 0.0)

        # --- Cornering speed limit (centripetal friction constraint) ----------
        # BendSegment and SlalomSegment both store v_max_corner.
        # For all other segments, no cornering limit applies.
        v_max_corner = getattr(segment, 'v_max_corner', float('inf'))

        # Overall speed cap: the lower of motor top speed and cornering limit
        max_speed        = min(self.no_load_speed, v_max_corner)
        corner_warned    = False   # Only warn once per segment

        speed    = 0.0
        distance = 0.0
        time     = 0.0

        while distance < segment.length:
            # Drive force from both motors
            f_drive = self.drive_force(speed)

            # Gravity component along slope (positive = opposing on uphill)
            f_gravity = self.gravity_force(angle_rad)

            # Rolling friction force (always opposes motion)
            f_friction = MU_ROLL * self.mass * GRAVITY_MS2 * math.cos(angle_rad)

            # Net force and resulting acceleration
            f_net = f_drive - f_gravity - f_friction
            accel = f_net / self.mass

            # Euler integration: update speed and position
            speed += accel * DT

            # --- Centripetal friction check -----------------------------------
            # If speed would exceed the cornering limit, the mouse would slide
            # off the track. Cap speed and warn once per curved segment.
            if speed > v_max_corner and not corner_warned:
                print(f"  ⚠ CORNERING WARNING on '{segment.name}': "
                      f"motor speed {speed:.3f} m/s exceeds friction limit "
                      f"{v_max_corner:.3f} m/s (μ={MU_LATERAL}, r={getattr(segment,'radius', getattr(segment,'bend_radius','?'))}m). "
                      f"Speed capped — mouse would leave track without speed control.")
                corner_warned = True

            speed     = max(0.0, min(speed, max_speed))
            distance += speed * DT
            time     += DT

            # Safety: break if stuck (uphill stall)
            if time > 300:
                print(f"  ⚠ STALL WARNING: Mouse stalled on '{segment.name}'!")
                break

        return time, speed


class MotorLapSimulator:
    """
    Stage 2: Physics-based lap time simulation using motor torque curves.

    Replaces the constant-speed assumption of Stage 1 with Newton's 2nd law
    applied to each segment in sequence. Speed carries over between segments
    (the mouse does not restart from zero each segment — only at the start
    of the lap).

    Attributes
    ----------
    motor : MotorModel
        The motor model to use.
    track : list of TrackSegment
        The ordered list of track segments.
    """

    def __init__(self, motor: MotorModel, track: list):
        """
        Initialise the Stage 2 simulator.

        Parameters
        ----------
        motor : MotorModel
            Configured motor model.
        track : list of TrackSegment
            Ordered track segments.
        """
        self.motor = motor
        self.track = track

    def simulate(self, verbose: bool = True) -> float:
        """
        Simulate the full lap and return the predicted time.

        Parameters
        ----------
        verbose : bool
            If True, print per-segment breakdown.

        Returns
        -------
        float
            Total predicted lap time in seconds.
        """
        print("\n" + "="*60)
        print("STAGE 2 — MOTOR & ACCELERATION MODEL")
        print(f"Mouse mass:       {self.motor.mass:.2f} kg")
        print(f"Motor max speed:  {self.motor.no_load_speed:.2f} m/s")
        print(f"Motor stall torq: {self.motor.stall_torque:.3f} N·m")
        print("="*60)

        total_time = 0.0

        for seg in self.track:
            seg_time, exit_speed = self.motor.simulate_segment(seg)
            total_time += seg_time

            if verbose:
                print(f"  {seg.name:<40} | "
                      f"{seg.length:.2f}m | "
                      f"exit speed={exit_speed:.2f}m/s | "
                      f"time={seg_time:.3f}s")

        print(f"\n  >>> Stage 2 Predicted Lap Time: {total_time:.3f} seconds\n")
        return total_time


# ===========================================================================
# STAGE 3 — PID DIFFERENTIAL STEERING MODEL
# ===========================================================================

class PIDController:
    """
    A discrete-time PID controller.

    Used to model the Arduino's closed-loop steering control. The controller
    output represents the speed differential applied between the left and
    right motors to steer the mouse back toward the track centre.

    Attributes
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    dt : float
        Time step in seconds.
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        """
        Initialise the PID controller.

        Parameters
        ----------
        kp : float
            Proportional gain.
        ki : float
            Integral gain.
        kd : float
            Derivative gain.
        dt : float
            Simulation time step in seconds.
        """
        self.kp         = kp
        self.ki         = ki
        self.kd         = kd
        self.dt         = dt
        self._integral  = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Reset integrator and previous error (call at start of each segment)."""
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float) -> float:
        """
        Compute the PID control output for a given error signal.

        Parameters
        ----------
        error : float
            The current tracking error (e.g. lateral offset from track centre).

        Returns
        -------
        float
            Control output (speed differential between left and right motors).
        """
        self._integral  += error * self.dt
        derivative       = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


class PIDLapSimulator:
    """
    Stage 3: Full differential-steering PID lap time simulation.

    Models the mouse as having separate left and right motor speeds. On
    straight sections, both motors run at equal speed. On bends and slaloms,
    the PID controller introduces a speed differential between the motors to
    follow the track, reducing the average forward speed.

    The lateral error is modelled as a sinusoidal disturbance scaled to the
    segment's curvature, representing the mouse oscillating around the track
    centre under closed-loop control.

    Attributes
    ----------
    pid : PIDController
        The PID controller instance.
    motor : MotorModel
        Motor physics model.
    track : list of TrackSegment
        Ordered track segments.
    """

    def __init__(self, pid: PIDController, motor: MotorModel, track: list):
        """
        Initialise the Stage 3 PID simulator.

        Parameters
        ----------
        pid : PIDController
            Configured PID controller.
        motor : MotorModel
            Configured motor model.
        track : list of TrackSegment
            Ordered track segments.
        """
        self.pid   = pid
        self.motor = motor
        self.track = track

    def _curvature_error_amplitude(self, segment: TrackSegment) -> float:
        """
        Estimate the lateral error amplitude induced by a segment's curvature.

        Straight segments produce no error. Bends and slaloms produce an
        error proportional to 1/radius (tighter bend = larger error).

        Parameters
        ----------
        segment : TrackSegment
            The current track segment.

        Returns
        -------
        float
            Amplitude of the sinusoidal lateral error signal in metres.
        """
        if isinstance(segment, BendSegment):
            return 0.05 / segment.radius  # Larger error for tighter radius
        elif isinstance(segment, SlalomSegment):
            return 0.08 / segment.bend_radius
        else:
            return 0.0  # No lateral disturbance on straights/ramps

    def simulate(self, verbose: bool = True) -> float:
        """
        Simulate the full lap with PID steering and return lap time.

        For each segment, the PID controller responds to the curvature-induced
        lateral error. The resulting speed differential reduces the average
        forward velocity of the mouse.

        Parameters
        ----------
        verbose : bool
            If True, print per-segment time breakdown.

        Returns
        -------
        float
            Total predicted lap time in seconds.
        """
        print("\n" + "="*60)
        print("STAGE 3 — PID DIFFERENTIAL STEERING MODEL")
        print(f"PID gains: Kp={self.pid.kp}, Ki={self.pid.ki}, Kd={self.pid.kd}")
        print("="*60)

        total_time = 0.0

        for seg in self.track:
            self.pid.reset()

            # Get error amplitude for this segment type
            error_amplitude = self._curvature_error_amplitude(seg)

            # --- Cornering speed limit (centripetal friction constraint) ------
            # Retrieve the physics-based cornering limit if this is a curved
            # segment. On straights and ramps there is no lateral limit.
            v_max_corner  = getattr(seg, 'v_max_corner', float('inf'))
            max_seg_speed = min(self.motor.no_load_speed, v_max_corner)
            corner_warned = False

            angle_rad  = getattr(seg, 'angle_rad', 0.0)
            distance   = 0.0
            time       = 0.0
            speed      = 0.0
            step       = 0

            while distance < seg.length:
                # Simulate sinusoidal lateral error (mouse oscillates around centre)
                lateral_error = error_amplitude * math.sin(2 * math.pi * step * DT)

                # PID output: speed differential between left and right motors
                delta_speed = self.pid.compute(lateral_error)

                # Average forward speed is reduced by the magnitude of the differential
                # (energy goes into turning rather than forward motion)
                speed_penalty = abs(delta_speed) * 0.3
                f_drive    = self.motor.drive_force(speed)
                f_gravity  = self.motor.gravity_force(angle_rad)
                f_friction = 0.05 * self.motor.mass * GRAVITY_MS2

                f_net = f_drive - f_gravity - f_friction
                accel = f_net / self.motor.mass

                speed += accel * DT

                # --- Centripetal friction check --------------------------------
                # Warn once if the PID-controlled speed would exceed the cornering
                # friction limit. The PID should prevent this in practice by
                # slowing the inner wheel — this warning flags if gains need tuning.
                if speed > v_max_corner and not corner_warned:
                    print(f"  ⚠ CORNERING WARNING (Stage 3) on '{seg.name}': "
                          f"speed {speed:.3f} m/s exceeds friction limit "
                          f"{v_max_corner:.3f} m/s. Check PID gains — "
                          f"mouse may leave track if gains cause overspeed on corners.")
                    corner_warned = True

                # Cap to the lower of motor speed and cornering limit, minus PID penalty
                speed     = max(0.0, min(speed, max_seg_speed - speed_penalty))
                distance += speed * DT
                time     += DT
                step     += 1

                if time > 300:
                    print(f"  ⚠ STALL WARNING: Stalled on '{seg.name}'!")
                    break

            total_time += time

            if verbose:
                print(f"  {seg.name:<40} | "
                      f"{seg.length:.2f}m | "
                      f"v_max_corner={v_max_corner:.2f}m/s | "
                      f"PID err={error_amplitude:.3f}m | "
                      f"time={time:.3f}s")

        print(f"\n  >>> Stage 3 Predicted Lap Time: {total_time:.3f} seconds\n")
        return total_time


# ===========================================================================
# MAIN — RUN ALL THREE STAGES
# ===========================================================================

def main():
    """
    Entry point: runs all three simulation stages and prints a comparison.

    Builds the track, configures the motor and PID models, runs each
    stage in sequence, and prints a summary comparison table.
    """
    print("\n" + "#"*60)
    print("#  MOUSE RACE LAP TIME SIMULATOR — EE22005 Project Week 3")
    print("#"*60)

    # --- Build the track ---
    track = build_track()

    # ------------------------------------------------------------------
    # STAGE 1: Kinematic model
    # Cruise speed estimated from motor no-load speed
    # ------------------------------------------------------------------
    cruise_speed = MOTOR_MAX_SPEED_MS
    stage1 = KinematicModel(cruise_speed=cruise_speed, track=track)
    t1 = stage1.simulate(verbose=True)

    # ------------------------------------------------------------------
    # STAGE 2: Motor & acceleration model
    # ------------------------------------------------------------------
    motor = MotorModel(
        stall_torque  = MOTOR_STALL_TORQUE_NM,
        no_load_speed = MOTOR_MAX_SPEED_MS,
        mass          = MOUSE_MASS_KG
    )
    stage2 = MotorLapSimulator(motor=motor, track=track)
    t2 = stage2.simulate(verbose=True)

    # ------------------------------------------------------------------
    # STAGE 3: PID differential steering model
    # ------------------------------------------------------------------
    pid = PIDController(kp=KP, ki=KI, kd=KD, dt=DT)
    stage3 = PIDLapSimulator(pid=pid, motor=motor, track=track)
    t3 = stage3.simulate(verbose=True)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"  Stage 1 (Kinematic only):          {t1:.3f} s")
    print(f"  Stage 2 (Motor + Acceleration):    {t2:.3f} s")
    print(f"  Stage 3 (PID Differential Drive):  {t3:.3f} s")
    print(f"  Course record:                      9.200 s")
    print("="*60)

    # Flag if estimated time beats or is close to the course record
    if t3 < 9.2:
        print("  ✓ Stage 3 prediction beats the course record!")
    elif t3 < 12.0:
        print("  ~ Stage 3 prediction is competitive (within 30% of record).")
    else:
        print("  ✗ Consider tuning motor voltage, PID gains, or mass.")


if __name__ == "__main__":
    main()
