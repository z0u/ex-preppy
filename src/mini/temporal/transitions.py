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
        if np.isclose(self.duration, 0.0):
            return self.final_value

        # Normalize time
        tau = t / self.duration

        # Clamp tau to [0, 1]
        tau = np.clip(tau, 0.0, 1.0)

        # If tau is 1 (or very close), ensure final value is exactly reached
        if np.isclose(tau, 1.0):
            return self.final_value

        # Calculate using the polynomial in tau
        c0, c1, c2, c3, c4, c5 = self.coeffs
        return c0 + c1 * tau + c2 * tau**2 + c3 * tau**3 + c4 * tau**4 + c5 * tau**5

    def get_state(self, t: float):
        """Get the value, velocity, and acceleration at time t."""
        if np.isclose(self.duration, 0.0):
            return self.initial_value, 0.0, 0.0

        # Normalize time
        tau = t / self.duration
        # Clamp tau to [0, 1]
        tau = np.clip(tau, 0.0, 1.0)

        c0, c1, c2, c3, c4, c5 = self.coeffs

        # Calculate value y(tau)
        value = c0 + c1 * tau + c2 * tau**2 + c3 * tau**3 + c4 * tau**4 + c5 * tau**5

        # Calculate derivative w.r.t. tau: y'(tau)
        dydtau = c1 + 2 * c2 * tau + 3 * c3 * tau**2 + 4 * c4 * tau**3 + 5 * c5 * tau**4

        # Calculate second derivative w.r.t. tau: y''(tau)
        d2ydtau2 = 2 * c2 + 6 * c3 * tau + 12 * c4 * tau**2 + 20 * c5 * tau**3

        # Scale derivatives back to be w.r.t. t
        velocity = dydtau / self.duration
        acceleration = d2ydtau2 / (self.duration**2)

        # If tau is 1 (or very close), ensure final state is exactly reached
        if np.isclose(tau, 1.0):
            value = self.final_value
            velocity = 0.0
            acceleration = 0.0

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
            self._value = self._interpolator.final_value
            self._interpolator = None
            self._ctime = self.duration

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

        if np.isclose(duration, 0.0):
            self._value = float(value)
            self._interpolator = None
            self._ctime = 0.0
            self._duration = 0.0  # Update the duration to be exactly 0.0
            return

        if self._interpolator is not None:
            current_value, current_velocity, current_acceleration = self._interpolator.get_state(self._ctime)
        else:
            current_value, current_velocity, current_acceleration = self._value, 0.0, 0.0

        if (
            np.isclose(current_value, value)
            and np.isclose(current_velocity, 0.0)
            and np.isclose(current_acceleration, 0.0)
        ):
            if self._value == value:
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
        self._duration = float(duration)

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

        if np.isclose(self._ctime, self.duration) or self._ctime > self.duration:
            return self._interpolator.final_value

        return self._interpolator(self._ctime)
