import pytest
import numpy as np
from src.utils.transitions import MinimumJerkTimingFunction, SmoothProp


class TestMinimumJerkTimingFunction:
    def test_initialization(self):
        # Test with default parameters
        mjf = MinimumJerkTimingFunction()
        assert mjf.initial_value == 0.0
        assert mjf.final_value == 0.0

        # Test with custom initial value
        mjf = MinimumJerkTimingFunction(initial_value=5.0)
        assert mjf.initial_value == 5.0
        assert mjf.final_value == 5.0

        # Test with both initial and final values
        mjf = MinimumJerkTimingFunction(initial_value=1.0, final_value=10.0)
        assert mjf.initial_value == 1.0
        assert mjf.final_value == 10.0

    def test_interpolation(self):
        # Create a function from 0 to 1
        mjf = MinimumJerkTimingFunction(initial_value=0.0, final_value=1.0)

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
        mjf = MinimumJerkTimingFunction(initial_value=0.0, initial_velocity=2.0, final_value=1.0)

        # With positive initial velocity, early values should exceed linear interpolation
        assert mjf(0.25) > 0.25

    def test_get_state(self):
        mjf = MinimumJerkTimingFunction(initial_value=0.0, final_value=1.0)

        # Test initial state
        value, velocity, acceleration = mjf.get_state(0.0)
        assert value == 0.0
        assert velocity == 0.0
        assert acceleration == 0.0

        # Test final state
        value, velocity, acceleration = mjf.get_state(1.0)
        assert pytest.approx(value, abs=1e-10) == 1.0
        assert pytest.approx(velocity, abs=1e-10) == 0.0
        assert pytest.approx(acceleration, abs=1e-10) == 0.0

        # Test intermediate state - all values should exist
        value, velocity, acceleration = mjf.get_state(0.5)
        assert 0.0 < value < 1.0
        assert isinstance(velocity, float)
        assert isinstance(acceleration, float)


class TestSmoothProp:
    def test_initialization(self):
        # Basic initialization
        prop = SmoothProp(5.0)
        assert prop._value == 5.0
        assert prop.value == 5.0
        assert prop.transition_duration == 0.0

        # With transition duration
        prop = SmoothProp(10.0, transition_duration=100.0)
        assert prop._value == 10.0
        assert prop.transition_duration == 100.0

    def test_immediate_transition(self):
        # With zero transition duration, changes should be immediate
        prop = SmoothProp(0.0, transition_duration=0.0)
        prop.value = 10.0
        assert prop.value == 10.0

        # Step should not affect the value
        prop.step(5.0)
        assert prop.value == 10.0

    def test_smooth_transition(self):
        # Create with transition duration
        prop = SmoothProp(0.0, transition_duration=10.0)

        # Change value to trigger transition
        prop.value = 1.0

        # At start, value should still be near initial value
        assert prop.value < 0.1

        # Step halfway through transition
        prop.step(5.0)
        mid_value = prop.value
        assert 0.0 < mid_value < 1.0

        # Complete transition
        prop.step(5.0)
        assert pytest.approx(prop.value, abs=1e-10) == 1.0

        # Ensure timing function is cleared after completion
        assert prop.interpolator is None

    def test_changing_target_mid_transition(self):
        prop = SmoothProp(0.0, transition_duration=10.0)

        # Start transition to 1.0
        prop.value = 1.0
        prop.step(2.0)

        # Value should have started moving toward 1.0
        assert 0.0 < prop.value < 1.0

        # Change target mid-transition (NOTE this resets ctime to 0)
        prop.value = 0.5

        # Complete the new transition (requires the full transition duration)
        prop.step(10.0)
        assert pytest.approx(prop.value, abs=1e-10) == 0.5

    def test_frac_property(self):
        prop = SmoothProp(0.0, transition_duration=10.0)

        assert prop.frac == 0.0

        prop.step(5.0)
        assert prop.frac == 0.5

        prop.step(10.0)  # Beyond duration
        assert prop.frac == 1.0
