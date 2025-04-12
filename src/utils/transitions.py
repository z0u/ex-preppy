import numpy as np


class MinimumJerkTimingFunction:
    """
    Implements a minimum jerk trajectory for smooth interpolation with guaranteed arrival time.

    Given a starting value at rest (zero velocity and acceleration), this function resembles the ease (cubic spline) timing function in CSS.

    This function also smoothly handles cases where the initial conditions are not at rest, allowing for more dynamic trajectories.
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        initial_velocity: float = 0.0,
        initial_acceleration: float = 0.0,
        final_value: float | None = None,
    ):
        """
        Initialize the interpolator with starting conditions.

        Args:
            initial_value: Starting value
            initial_velocity: Starting velocity (rate of change)
            initial_acceleration: Starting acceleration (optional, default 0)
            final_value: Target final value (optional, default None)
        """
        self.initial_value = initial_value
        self.initial_velocity = initial_velocity
        self.initial_acceleration = initial_acceleration

        self.final_value = final_value if final_value is not None else self.initial_value
        self.coeffs = self._calculate_coefficients(
            y0=self.initial_value,
            v0=self.initial_velocity,
            a0=self.initial_acceleration,
            y1=self.final_value,
            v1=0.0,  # Target velocity (typically zero)
            a1=0.0,  # Target acceleration (typically zero)
            T=1.0,  # Duration (normalized to 1.0)
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

    def __call__(self, frac: float):
        """
        Get the interpolated value at a specific time without changing internal state.

        Args:
            frac: Fraction of the total duration to evaluate (0 to 1)

        Returns:
            The value at time t
        """
        # Clamp to duration
        if frac >= 1.0:
            return self.final_value

        # Calculate using the polynomial
        c0, c1, c2, c3, c4, c5 = self.coeffs
        return c0 + c1 * frac + c2 * frac**2 + c3 * frac**3 + c4 * frac**4 + c5 * frac**5

    def get_state(self, frac: float = 0.0):
        c0, c1, c2, c3, c4, c5 = self.coeffs
        value = c0 + c1 * frac + c2 * frac**2 + c3 * frac**3 + c4 * frac**4 + c5 * frac**5

        # Calculate velocity (first derivative)
        velocity = c1 + 2 * c2 * frac + 3 * c3 * frac**2 + 4 * c4 * frac**3 + 5 * c5 * frac**4

        # Calculate acceleration (second derivative)
        acceleration = 2 * c2 + 6 * c3 * frac + 12 * c4 * frac**2 + 20 * c5 * frac**3

        return value, velocity, acceleration


class SmoothProp:
    """Stores a value, and smoothly transitions new value on change."""

    interpolator: MinimumJerkTimingFunction | None

    def __init__(self, value: float, transition_duration: float = 0.0):
        self.transition_duration = transition_duration
        self._value = float(value)
        self.interpolator = None
        self.ctime = 0.0

    def step(self, n=1.0):
        self.ctime += float(n)

    @property
    def transition_duration(self):
        return self._transition_duration

    @transition_duration.setter
    def transition_duration(self, duration: float):
        """Set the transition duration."""
        if np.isclose(duration, 0.0):
            duration = 0.0
        if duration < 0.0:
            raise ValueError("Transition duration must be non-negative.")
        self._transition_duration = duration

    @property
    def frac(self):
        # Return 1.0 when transition_duration is 0 to avoid division by zero
        if np.isclose(self.transition_duration, 0.0):
            return 1.0

        frac = self.ctime / self.transition_duration
        if np.isclose(frac, 1.0) or frac > 1.0:
            return 1.0
        if np.isclose(frac, 0.0) or frac < 0.0:
            return 0.0
        return frac

    @property
    def value(self):
        """Get the smoothed value, according to the timing function."""
        if self.frac >= 1.0:
            self.interpolator = None
        if not self.interpolator:
            return self._value
        return self.interpolator(self.frac)

    @value.setter
    def value(self, new_value: float):
        """Set the target value."""
        if new_value != self._value and self.transition_duration > 0:
            if self.interpolator is None:
                value, velocity, acceleration = self._value, 0.0, 0.0
            else:
                value, velocity, acceleration = self.interpolator.get_state(self.frac)
            self.interpolator = MinimumJerkTimingFunction(
                initial_value=value,
                initial_velocity=velocity,
                initial_acceleration=acceleration,
                final_value=new_value,
            )
        self.ctime = 0.0
        self._value = new_value
