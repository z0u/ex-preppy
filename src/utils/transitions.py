import numpy as np


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
        Calculate the coefficients for the 5th-degree polynomial.

        Args:
            y0: Initial position
            v0: Initial velocity
            a0: Initial acceleration
            y1: Target position
            v1: Target velocity (typically 0)
            a1: Target acceleration (typically 0)
            T: Duration

        Returns:
            List of 6 coefficients [c0, c1, c2, c3, c4, c5]
        """
        # First 3 coefficients are determined by initial conditions
        c0 = y0
        c1 = v0
        c2 = a0 / 2.0

        # Set up the system of equations for the remaining coefficients
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        # Right-hand side of the equation
        b1 = y1 - y0 - v0 * T - (a0 / 2.0) * T2
        b2 = v1 - v0 - a0 * T
        b3 = a1 - a0

        # Coefficient matrix
        A = np.array(
            [
                [T3, T4, T5],
                [3 * T2, 4 * T3, 5 * T4],
                [6 * T, 12 * T2, 20 * T3],
            ],
        )

        b = np.array([b1, b2, b3])

        # Solve the system of equations
        x = np.linalg.solve(A, b)

        # Extract the coefficients
        c3, c4, c5 = x

        return [c0, c1, c2, c3, c4, c5]

    def __call__(self, t: float):
        """
        Get the interpolated value at a specific time without changing internal state.

        Args:
            t: Time to evaluate (0 to duration)

        Returns:
            The value at time t
        """
        # Clamp to duration
        if t >= self.duration:
            return self.final_value

        # Calculate using the polynomial
        c0, c1, c2, c3, c4, c5 = self.coeffs
        return c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5

    def get_state(self, t: float):
        c0, c1, c2, c3, c4, c5 = self.coeffs
        value = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5

        # Calculate velocity (first derivative)
        velocity = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4

        # Calculate acceleration (second derivative)
        acceleration = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3

        return value, velocity, acceleration


class SmoothProp:
    """Stores a value, and smoothly transitions new value on change."""

    _interpolator: MinimumJerkTimingFunction | None
    _ctime: float
    """The number of steps that have been taken since the last set() call."""
    _duration: float
    """The number of steps to take to reach the target value."""
    _value: float
    """The target value"""

    def __init__(self, value: float, duration: float = 0.0):
        """
        Initialize the SmoothProp with a starting value and optional duration.

        Args:
            value: Initial value
            duration: Duration of the transition (unitless; see `step()`)
        """
        self._duration = duration
        self._value = float(value)
        self._interpolator = None
        self._ctime = 0.0

    def step(self, n=1.0):
        """
        Progress the internal clock forward.

        Args:
            n: Amount to step the clock forward. This is a unitless value, and could
            mean "frames" or some real amount of time. For example, if called with the
            elapsed real time in seconds, the duration would be in seconds.
        """
        self._ctime += float(n)
        # Clear the interpolator if we've reached or exceeded the duration
        if self._interpolator is not None and (np.isclose(self._ctime, self.duration) or self._ctime > self.duration):
            self._interpolator = None

    def set(self, value: float | None = None, duration: float | None = None):
        """
        Set a new target value and/or duration for the transition.

        Args:
            value: Target value to transition to. If `None`, keeps the current value.
            duration: Duration of the transition. If `None`, keeps the current duration.

        Always resets the internal clock, therefore, the target value may only be
        reached after the specified duration — even if the current value is close.
        Depending on the current velocity, it may also overshoot (and come back).
        """
        value = value if value is not None else self._value
        duration = duration if duration is not None else self._duration

        if duration <= 0.0:
            # If duration is 0, just set the value immediately
            self._value = float(value)
            self._interpolator = None
            self._ctime = 0.0
            return

        if self._interpolator is not None:
            current_value, current_velocity, current_acceleration = self._interpolator.get_state(self._ctime)
        else:
            current_value, current_velocity, current_acceleration = self._value, 0.0, 0.0

        if current_value == value and current_velocity == 0.0 and current_acceleration == 0.0:
            # Already reached steady state
            return

        self._interpolator = MinimumJerkTimingFunction(
            initial_value=current_value,
            initial_velocity=current_velocity,
            initial_acceleration=current_acceleration,
            final_value=value,
            duration=duration,
        )
        self._ctime = 0.0
        self._value = float(value)

    @property
    def duration(self):
        """
        Get the duration of the transition.

        This is a unitless value. The actual time it takes depends on how `step()` is
        called.
        """
        return self._duration

    @property
    def value(self):
        """Get the smoothed value, according to the timing function."""
        if self._interpolator is None:
            return self._value

        # Clamp time to duration
        if np.isclose(self._ctime, self.duration) or self._ctime > self.duration:
            return self._value

        return self._interpolator(self._ctime)
