import numpy as np
import pytest

from mini.temporal.model import DynamicPropState
from mini.temporal.timing_fn import (
    LinearTimingFunction,
    MinimumJerkTimingFunction,
    StepEndTimingFunction,
)


class TestMinimumJerkTimingFunction:
    def test_interpolation(self):
        # Create a function from 0 to 1
        mjf = MinimumJerkTimingFunction(
            initial_value=0.0,
            final_value=1.0,
            duration=1.0,
            initial_velocity=0.0,
            initial_acceleration=0.0,
        )

        # At t=0, should be at initial value
        assert mjf(0.0) == 0.0

        # At t=1, should be at final value
        assert mjf(1.0) == 1.0

        # At t=0.5, should be somewhere between (not linear)
        mid_value = mjf(0.5)
        assert 0.0 < mid_value < 1.0

        # Test t > 1.0 (should clamp to final value)
        assert mjf(1.5) == 1.0

    def test_with_velocity(self):
        # Test with initial velocity
        mjf = MinimumJerkTimingFunction(
            initial_value=0.0, initial_velocity=2.0, initial_acceleration=0.0, final_value=1.0, duration=1.0
        )

        # With positive initial velocity, early values should exceed linear interpolation
        assert mjf(0.25) > 0.25

    def test_get_state(self):
        mjf = MinimumJerkTimingFunction(
            initial_value=0.0, final_value=1.0, duration=1.0, initial_velocity=0.0, initial_acceleration=0.0
        )

        # Test initial state
        assert mjf.get_state(0.0) == pytest.approx(DynamicPropState(value=0.0, velocity=0.0, acceleration=0.0))

        # Test final state
        assert mjf.get_state(1.0) == pytest.approx(
            DynamicPropState(value=1.0, velocity=0.0, acceleration=0.0), abs=1e-9
        )

        # Test intermediate state - all values should exist
        state = mjf.get_state(0.5)
        assert isinstance(state, DynamicPropState)
        assert 0.0 < state.value < 1.0
        assert isinstance(state.velocity, float)
        assert isinstance(state.acceleration, float)

    def test_long_duration(self):
        """Test interpolation with a very long duration to check for overflow."""
        long_duration = 20000.0
        mjf = MinimumJerkTimingFunction(
            initial_value=0.0,
            final_value=1.0,
            duration=long_duration,
            initial_velocity=0.0,
            initial_acceleration=0.0,
        )

        # Check initial state
        assert mjf(0.0) == 0.0
        assert mjf.get_state(0.0) == pytest.approx(DynamicPropState(value=0.0, velocity=0.0, acceleration=0.0))

        # Check intermediate state (should not be NaN or inf)
        mid_value = mjf(long_duration / 2)
        assert np.isfinite(mid_value)
        assert 0.0 < mid_value < 1.0

        # Check final state
        assert pytest.approx(mjf(long_duration), abs=1e-9) == 1.0
        assert mjf.get_state(long_duration) == pytest.approx(
            DynamicPropState(value=1.0, velocity=0.0, acceleration=0.0), abs=1e-9
        )

    def test_zero_duration(self):
        """Test behavior when duration is zero."""
        initial_val = 5.0
        final_val = 10.0  # This should be ignored, final = initial if duration is 0

        mjf = MinimumJerkTimingFunction(
            initial_value=initial_val,
            final_value=final_val,
            duration=0.0,
            initial_velocity=1.0,  # Should be ignored
            initial_acceleration=2.0,  # Should be ignored
        )

        # Value should immediately be the initial value
        assert mjf(0.0) == initial_val
        assert mjf(10.0) == initial_val  # Any time t should return initial_val

        # State should be constant
        expected_state = DynamicPropState(value=initial_val, velocity=0.0, acceleration=0.0)
        assert mjf.get_state(0.0) == pytest.approx(expected_state)
        assert mjf.get_state(100.0) == pytest.approx(expected_state)

        # Check final_value was updated
        assert mjf.final_value == initial_val


class TestLinearTimingFunction:
    def test_interpolation(self):
        """Test basic linear interpolation."""
        func = LinearTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=0, initial_acceleration=0
        )
        assert func(0.0) == pytest.approx(0.0)
        assert func(5.0) == pytest.approx(5.0)
        assert func(10.0) == pytest.approx(10.0)
        assert func(11.0) == pytest.approx(10.0)  # Clamp past duration
        assert func(-1.0) == pytest.approx(0.0)  # Clamp before start

    def test_get_state(self):
        """Test getting value, velocity, and acceleration."""
        func = LinearTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=0, initial_acceleration=0
        )
        # Start
        assert func.get_state(0.0) == pytest.approx(DynamicPropState(value=0.0, velocity=1.0, acceleration=0.0))
        # Mid
        assert func.get_state(5.0) == pytest.approx(DynamicPropState(value=5.0, velocity=1.0, acceleration=0.0))
        # End
        assert func.get_state(10.0) == pytest.approx(DynamicPropState(value=10.0, velocity=1.0, acceleration=0.0))
        # Past End (state clamps like value)
        assert func.get_state(11.0) == pytest.approx(DynamicPropState(value=10.0, velocity=1.0, acceleration=0.0))

    def test_zero_duration(self):
        """Test behavior with zero duration."""
        func = LinearTimingFunction(
            initial_value=5.0, final_value=10.0, duration=0.0, initial_velocity=1, initial_acceleration=1
        )
        assert func(0.0) == pytest.approx(5.0)
        assert func(1.0) == pytest.approx(5.0)
        expected_state = DynamicPropState(value=5.0, velocity=0.0, acceleration=0.0)
        assert func.get_state(0.0) == pytest.approx(expected_state)
        assert func.final_value == pytest.approx(5.0)  # Final value updated

    def test_ignores_initial_velocity_acceleration(self):
        """Test that initial velocity/acceleration are ignored."""
        func = LinearTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=100.0, initial_acceleration=100.0
        )
        # State should be the same as if initial vel/accel were 0
        assert func.get_state(0.0) == pytest.approx(DynamicPropState(value=0.0, velocity=1.0, acceleration=0.0))
        assert func(5.0) == pytest.approx(5.0)  # Midpoint value unaffected


class TestStepEndTimingFunction:
    def test_interpolation(self):
        """Test step-end interpolation."""
        func = StepEndTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=0, initial_acceleration=0
        )
        assert func(-1.0) == pytest.approx(0.0)  # Clamp before start
        assert func(0.0) == pytest.approx(0.0)
        assert func(5.0) == pytest.approx(0.0)
        assert func(9.999) == pytest.approx(0.0)
        assert func(10.0) == pytest.approx(10.0)  # Jumps at t == duration
        assert func(11.0) == pytest.approx(10.0)  # Stays at final value

    def test_get_state(self):
        """Test getting value, velocity, and acceleration."""
        func = StepEndTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=0, initial_acceleration=0
        )
        # Before jump
        assert func.get_state(5.0) == pytest.approx(DynamicPropState(value=0.0, velocity=0.0, acceleration=0.0))
        # At jump point (state reflects the *new* value)
        assert func.get_state(10.0) == pytest.approx(DynamicPropState(value=10.0, velocity=0.0, acceleration=0.0))
        # After jump
        assert func.get_state(11.0) == pytest.approx(DynamicPropState(value=10.0, velocity=0.0, acceleration=0.0))

    def test_zero_duration(self):
        """Test behavior with zero duration."""
        func = StepEndTimingFunction(
            initial_value=5.0, final_value=10.0, duration=0.0, initial_velocity=1, initial_acceleration=1
        )
        assert func(0.0) == pytest.approx(5.0)  # Immediate jump to initial
        assert func(1.0) == pytest.approx(5.0)
        expected_state = DynamicPropState(value=5.0, velocity=0.0, acceleration=0.0)
        assert func.get_state(0.0) == pytest.approx(expected_state)
        assert func.final_value == pytest.approx(5.0)  # Final value updated

    def test_ignores_initial_velocity_acceleration(self):
        """Test that initial velocity/acceleration are ignored."""
        func = StepEndTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=100.0, initial_acceleration=100.0
        )
        # State should be the same as if initial vel/accel were 0
        assert func.get_state(5.0) == pytest.approx(DynamicPropState(value=0.0, velocity=0.0, acceleration=0.0))
        assert func(10.0) == pytest.approx(10.0)  # Jump value unaffected
