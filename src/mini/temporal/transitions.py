import numpy as np
from typing import Protocol, Type
from dataclasses import dataclass
import math


@dataclass
class TimingState:
    """Represents the state of a property at a given time."""
    value: float
    velocity: float
    acceleration: float


class TimingFunction(Protocol):
    """Protocol for timing functions used by SmoothProp."""

    initial_value: float
    final_value: float
    duration: float

    def __init__(
        self,
        initial_value: float,
        initial_velocity: float,
        initial_acceleration: float,
        final_value: float,
        duration: float,
    ):
        """Initialize the timing function with initial state and target."""
        ...

    def __call__(self, t: float) -> float:
        """Get the interpolated value at time t."""
        ...

    def get_state(self, t: float) -> TimingState:
        """Get the value, velocity, and acceleration at time t."""
        ...


class MinimumJerkTimingFunction:
    """
    Implements a minimum jerk trajectory for smooth interpolation with guaranteed arrival time.

    Given a starting value at rest (zero velocity and acceleration), this function resembles the ease (cubic spline) timing function in CSS.

    This function also smoothly handles cases where the initial conditions are not at rest, allowing for more dynamic trajectories.
    """

    def __init__(
        self,
        initial_value: float,
        initial_velocity: float,
        initial_acceleration: float,
        final_value: float,
        duration: float,
    ):
        """
        Initialize the interpolator with starting conditions.

        Args:
            initial_value: Starting value
            initial_velocity: Starting velocity (rate of change)
            initial_acceleration: Starting acceleration
            final_value: Target final value
            duration: Duration of the transition (unitless)
        """
        self.initial_value = initial_value
        self.initial_velocity = initial_velocity
        self.initial_acceleration = initial_acceleration

        self.final_value = final_value
        self.duration = duration
        # Handle zero duration case to avoid division by zero
        if np.isclose(duration, 0.0):
            self.coeffs = [initial_value, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.final_value = initial_value
        else:
            self.coeffs = self._calculate_coefficients(
                y0=self.initial_value,
                v0=self.initial_velocity,
                a0=self.initial_acceleration,
                y1=self.final_value,
                v1=0.0,  # Target velocity (typically zero)
                a1=0.0,  # Target acceleration (typically zero)
                T=self.duration,
            )

    def _calculate_coefficients(self, y0, v0, a0, y1, v1, a1, T):
        """
        Calculate the coefficients for the 5th-degree polynomial using normalized time tau = t / T.

        Args:
            y0: Initial position
            v0: Initial velocity
            a0: Initial acceleration
            y1: Target position
            v1: Target velocity (typically 0)
            a1: Target acceleration (typically 0)
            T: Duration

        Returns:
            List of 6 coefficients [c0, c1, c2, c3, c4, c5] for y(tau)
        """
        # Coefficients based on initial conditions at tau = 0
        c0 = y0
        c1 = v0 * T  # Scaled initial velocity: dy/dtau = dy/dt * dt/dtau = v0 * T
        c2 = a0 * T**2 / 2.0  # Scaled initial acceleration: d^2y/dtau^2 = d^2y/dt^2 * (dt/dtau)^2 = a0 * T^2

        # System of equations for c3, c4, c5 based on final conditions at tau = 1
        # Matrix A is constant for the normalized system
        A = np.array(
            [
                [1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0],
                [6.0, 12.0, 20.0],
            ],
        )

        # Right-hand side vector b, incorporating scaled target velocity and acceleration
        b1 = y1 - c0 - c1 - c2
        b2 = v1 * T - c1 - 2 * c2  # Scaled target velocity
        b3 = a1 * T**2 - 2 * c2  # Scaled target acceleration

        b = np.array([b1, b2, b3])

        # Solve the system A * x = b for x = [c3, c4, c5]
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print(f'Warning: Linear algebra solver failed for T={T}. Using fallback coefficients.')  # noqa: T201
            c3 = y1 - c0 - c1 - c2
            c4 = 0.0
            c5 = 0.0
            x = np.array([c3, c4, c5])

        c3, c4, c5 = x

        return [c0, c1, c2, c3, c4, c5]

    def __call__(self, t: float):
        """
        Get the interpolated value at a specific time t.

        Args:
            t: Time to evaluate (0 to duration)

        Returns:
            The value at time t
        """
        # Direct comparison is much faster than np.isclose for scalars
        if -0.0000000001 < self.duration < 0.0000000001:  # Equivalent to np.isclose with default tolerance
            return self.final_value

        # Normalize time
        tau = t / self.duration

        # Code hotspot! Manual optimization. Faster than np.clip for scalars
        if tau < 0.0:
            tau = 0.0
        elif tau > 1.0:
            tau = 1.0

        # Code hotspot with manual optimization
        if 0.9999999999 < tau < 1.0000000001:
            return self.final_value

        # Calculate using the polynomial in tau
        c0, c1, c2, c3, c4, c5 = self.coeffs
        # Using Horner's method instead of using ** (optimization)
        return c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * c5))))

    def get_state(self, t: float) -> TimingState:
        """Get the value, velocity, and acceleration at time t."""
        if -0.0000000001 < self.duration < 0.0000000001:  # Equivalent to np.isclose with default tolerance
            return TimingState(value=self.initial_value, velocity=0.0, acceleration=0.0)

        # Normalize time
        tau = t / self.duration

        # Replace np.clip
        tau = max(0.0, min(1.0, tau))

        c0, c1, c2, c3, c4, c5 = self.coeffs

        # Calculate value y(tau)
        # Using Horner's method instead of using ** (optimization)
        value = c0 + tau * (c1 + tau * (c2 + tau * (c3 + tau * (c4 + tau * c5))))

        # Calculate derivative w.r.t. tau: y'(tau) using Horner's method
        dydtau = c1 + tau * (2 * c2 + tau * (3 * c3 + tau * (4 * c4 + tau * 5 * c5)))

        # Calculate second derivative w.r.t. tau: y''(tau) using Horner's method
        d2ydtau2 = 2 * c2 + tau * (6 * c3 + tau * (12 * c4 + tau * 20 * c5))

        # Scale derivatives back to be w.r.t. t
        if abs(self.duration) < 1e-12:
            velocity = 0.0
            acceleration = 0.0
        else:
            velocity = dydtau / self.duration
            acceleration = d2ydtau2 / (self.duration**2)

        # Replace np.isclose
        if 0.9999999999 < tau < 1.0000000001:
            value = self.final_value
            velocity = 0.0
            acceleration = 0.0

        return TimingState(value=value, velocity=velocity, acceleration=acceleration)


class LinearTimingFunction:
    """Implements simple linear interpolation."""
    initial_value: float
    final_value: float
    duration: float
    _velocity: float

    def __init__(
        self,
        initial_value: float,
        initial_velocity: float,  # Ignored
        initial_acceleration: float,  # Ignored
        final_value: float,
        duration: float,
    ):
        self.initial_value = initial_value
        self.duration = duration

        if np.isclose(duration, 0.0):
            self.final_value = initial_value  # Jump immediately
            self._velocity = 0.0
        else:
            self.final_value = final_value
            self._velocity = (final_value - initial_value) / duration

    def __call__(self, t: float) -> float:
        if np.isclose(self.duration, 0.0):
            return self.initial_value

        # Clamp time
        t = max(0.0, min(self.duration, t))

        # Linear interpolation formula: y = y0 + t * (y1 - y0) / T
        # Simplified: y = y0 + t * velocity
        return self.initial_value + t * self._velocity

    def get_state(self, t: float) -> TimingState:
        value = self(t)  # Calculate value using __call__
        # Clamp state at the end
        if t >= self.duration and not np.isclose(self.duration, 0.0):
            return TimingState(value=self.final_value, velocity=self._velocity, acceleration=0.0)
        # Clamp state at the beginning
        if t < 0.0:
            return TimingState(value=self.initial_value, velocity=self._velocity, acceleration=0.0)

        # For zero duration, state is fixed at initial
        if np.isclose(self.duration, 0.0):
            return TimingState(value=self.initial_value, velocity=0.0, acceleration=0.0)

        return TimingState(value=value, velocity=self._velocity, acceleration=0.0)


class StepEndTimingFunction:
    """Holds the initial value until the duration is reached, then jumps to the final value."""
    initial_value: float
    final_value: float
    duration: float

    def __init__(
        self,
        initial_value: float,
        initial_velocity: float,  # Ignored
        initial_acceleration: float,  # Ignored
        final_value: float,
        duration: float,
    ):
        self.initial_value = initial_value
        self.duration = duration
        # If duration is zero, the "final" value is actually the initial one
        self.final_value = initial_value if np.isclose(duration, 0.0) else final_value

    def __call__(self, t: float) -> float:
        if np.isclose(self.duration, 0.0):
            return self.initial_value  # Always initial if duration is zero

        # Use >= for step-end behavior (jump happens *at* t == duration)
        if t >= self.duration:
            return self.final_value
        elif t < 0.0:
            return self.initial_value
        else:
            return self.initial_value

    def get_state(self, t: float) -> TimingState:
        value = self(t)
        # Velocity and acceleration are always zero for step functions
        return TimingState(value=value, velocity=0.0, acceleration=0.0)


class SmoothProp:
    """Stores a value, and smoothly transitions new value on change."""

    _interpolator: TimingFunction | None
    _timing_function_cls: Type[TimingFunction]
    _ctime: float
    """The number of steps that have been taken since the last set() call."""
    _duration: float
    """The number of steps to take to reach the target value."""
    _value: float
    """The target value"""

    def __init__(
        self,
        value: float,
        duration: float = 0.0,
        timing_function_cls: Type[TimingFunction] = MinimumJerkTimingFunction,
    ):
        """
        Initialize the SmoothProp with a starting value and optional duration.

        Args:
            value: Initial value
            duration: Duration of the transition (unitless; see `step()`)
            timing_function_cls: The class to use for interpolation.
        """
        self._timing_function_cls = timing_function_cls
        self._duration = duration
        self._value = float(value)
        self._interpolator = None
        self._ctime = 0.0
        if not np.isclose(duration, 0.0):
            self._interpolator = self._timing_function_cls(
                initial_value=value,
                initial_velocity=0.0,
                initial_acceleration=0.0,
                final_value=value,
                duration=duration,
            )

    def step(self, n=1.0):
        """
        Progress the internal clock forward.

        Args:
            n: Amount to step the clock forward. This is a unitless value, and could
            mean "frames" or some real amount of time. For example, if called with the
            elapsed real time in seconds, the duration would be in seconds.
        """
        if self._interpolator is None:
            return

        self._ctime += float(n)
        if abs(self._ctime - self.duration) < 1e-10 or self._ctime > self.duration:
            final_val = self._interpolator.final_value
            self._interpolator = None
            self._value = final_val
            self._ctime = self.duration

    def set(self, value: float | None = None, duration: float | None = None):
        """
        Set a new target value and/or duration for the transition.

        Args:
            value: Target value to transition to. If `None`, keeps the current target value.
            duration: Duration of the transition. If `None`, keeps the current duration.

        Starts a new transition from the current state (value, velocity, acceleration)
        towards the new target value over the specified duration, using the configured
        timing function. Resets the internal clock `_ctime`.
        """
        new_target_value = value if value is not None else (self._interpolator.final_value if self._interpolator else self._value)
        new_duration = duration if duration is not None else self._duration

        # Get current state before potentially overwriting interpolator
        if self._interpolator is not None:
            # Ensure ctime is clamped before getting state if it overshot
            clamped_ctime = max(0.0, min(self._interpolator.duration, self._ctime))
            current_state = self._interpolator.get_state(clamped_ctime) # Get TimingState object
        else:
            # If no interpolator, assume we are at rest at the last set value
            current_state = TimingState(value=self._value, velocity=0.0, acceleration=0.0) # Create TimingState

        # Handle immediate jump (zero duration)
        if np.isclose(new_duration, 0.0):
            self._value = float(new_target_value)
            self._interpolator = None
            self._ctime = 0.0
            self._duration = 0.0
            return

        # Avoid creating a new interpolator if the target hasn't changed
        # and we are already at rest at the target.
        if (
            self._interpolator is None # Already at rest
            and np.isclose(current_state.value, new_target_value) # Check value from state
        ):
             if not np.isclose(self._duration, new_duration):
                 self._duration = float(new_duration)
             return # No change needed

        # Avoid creating a new interpolator if target value and duration are identical to current transition
        # This prevents redundant object creation if set() is called repeatedly with the same target.
        if (
            self._interpolator is not None
            and np.isclose(self._interpolator.final_value, new_target_value)
            and np.isclose(self._interpolator.duration, new_duration)
        ):
             # Update the internal target value and duration in case they were None
             self._value = float(new_target_value)
             self._duration = float(new_duration)
             # Don't reset _ctime or create a new interpolator
             return


        # Create the new interpolator using the stored class
        self._interpolator = self._timing_function_cls(
            initial_value=current_state.value,         # Unpack from state
            initial_velocity=current_state.velocity,   # Unpack from state
            initial_acceleration=current_state.acceleration, # Unpack from state
            final_value=new_target_value,
            duration=new_duration,
        )
        self._ctime = 0.0 # Reset clock for new transition
        self._value = float(new_target_value) # Update target value
        self._duration = float(new_duration) # Update duration

    @property
    def duration(self):
        """
        Get the duration of the transition.

        This is a unitless value. The actual time it takes depends on how `step()` is
        called.
        """
        return self._interpolator.duration if self._interpolator is not None else self._duration

    @property
    def value(self):
        """Get the smoothed value, according to the timing function."""
        if self._interpolator is None:
            return self._value

        if abs(self._ctime - self.duration) < 1e-10 or self._ctime > self.duration:
            return self._interpolator.final_value

        return self._interpolator(self._ctime)


class LogSpaceSmoothProp(SmoothProp):
    """A wrapper for SmoothProp that operates in logarithmic space."""

    def __init__(
        self,
        value: float,
        duration: float = 0.0,
        timing_function_cls: Type[TimingFunction] = MinimumJerkTimingFunction,
    ):
        """Initialize with a value in normal space and convert to log space internally."""
        # Ensure value is positive for log space
        safe_value = max(value, 1e-10)
        log_value = math.log(safe_value)

        # Initialize the parent SmoothProp in log space
        super().__init__(
            value=log_value,
            duration=duration,
            timing_function_cls=timing_function_cls,
        )

    def step(self, n=1.0):
        """Progress the internal clock forward."""
        super().step(n)

    def set(self, value: float | None = None, duration: float | None = None):
        """Set a new target value and/or duration for the transition."""
        # If value is provided, convert to log space
        if value is not None:
            # Ensure value is positive for log space
            safe_value = max(value, 1e-10)
            log_value = math.log(safe_value)
            super().set(value=log_value, duration=duration)
        else:
            super().set(value=None, duration=duration)

    @property
    def value(self):
        """Get the current value, transformed back from log space."""
        # Get the log-space value from the parent class
        log_value = super().value
        # Transform back to normal space
        return math.exp(log_value)
