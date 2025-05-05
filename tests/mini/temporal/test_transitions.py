import numpy as np
import pytest
from mini.temporal.transitions import (
    LinearTimingFunction,
    MinimumJerkTimingFunction,
    SmoothProp,
    StepEndTimingFunction,
    TimingState,
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
        assert mjf.get_state(0.0) == pytest.approx(TimingState(value=0.0, velocity=0.0, acceleration=0.0))

        # Test final state
        assert mjf.get_state(1.0) == pytest.approx(TimingState(value=1.0, velocity=0.0, acceleration=0.0), abs=1e-9)

        # Test intermediate state - all values should exist
        state = mjf.get_state(0.5)
        assert isinstance(state, TimingState)
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
        assert mjf.get_state(0.0) == pytest.approx(TimingState(value=0.0, velocity=0.0, acceleration=0.0))

        # Check intermediate state (should not be NaN or inf)
        mid_value = mjf(long_duration / 2)
        assert np.isfinite(mid_value)
        assert 0.0 < mid_value < 1.0

        # Check final state
        assert pytest.approx(mjf(long_duration), abs=1e-9) == 1.0
        assert mjf.get_state(long_duration) == pytest.approx(
            TimingState(value=1.0, velocity=0.0, acceleration=0.0), abs=1e-9
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
        expected_state = TimingState(value=initial_val, velocity=0.0, acceleration=0.0)
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
        assert func.get_state(0.0) == pytest.approx(TimingState(value=0.0, velocity=1.0, acceleration=0.0))
        # Mid
        assert func.get_state(5.0) == pytest.approx(TimingState(value=5.0, velocity=1.0, acceleration=0.0))
        # End
        assert func.get_state(10.0) == pytest.approx(TimingState(value=10.0, velocity=1.0, acceleration=0.0))
        # Past End (state clamps like value)
        assert func.get_state(11.0) == pytest.approx(TimingState(value=10.0, velocity=1.0, acceleration=0.0))

    def test_zero_duration(self):
        """Test behavior with zero duration."""
        func = LinearTimingFunction(
            initial_value=5.0, final_value=10.0, duration=0.0, initial_velocity=1, initial_acceleration=1
        )
        assert func(0.0) == pytest.approx(5.0)
        assert func(1.0) == pytest.approx(5.0)
        expected_state = TimingState(value=5.0, velocity=0.0, acceleration=0.0)
        assert func.get_state(0.0) == pytest.approx(expected_state)
        assert func.final_value == pytest.approx(5.0)  # Final value updated

    def test_ignores_initial_velocity_acceleration(self):
        """Test that initial velocity/acceleration are ignored."""
        func = LinearTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=100.0, initial_acceleration=100.0
        )
        # State should be the same as if initial vel/accel were 0
        assert func.get_state(0.0) == pytest.approx(TimingState(value=0.0, velocity=1.0, acceleration=0.0))
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
        assert func.get_state(5.0) == pytest.approx(TimingState(value=0.0, velocity=0.0, acceleration=0.0))
        # At jump point (state reflects the *new* value)
        assert func.get_state(10.0) == pytest.approx(TimingState(value=10.0, velocity=0.0, acceleration=0.0))
        # After jump
        assert func.get_state(11.0) == pytest.approx(TimingState(value=10.0, velocity=0.0, acceleration=0.0))

    def test_zero_duration(self):
        """Test behavior with zero duration."""
        func = StepEndTimingFunction(
            initial_value=5.0, final_value=10.0, duration=0.0, initial_velocity=1, initial_acceleration=1
        )
        assert func(0.0) == pytest.approx(5.0)  # Immediate jump to initial
        assert func(1.0) == pytest.approx(5.0)
        expected_state = TimingState(value=5.0, velocity=0.0, acceleration=0.0)
        assert func.get_state(0.0) == pytest.approx(expected_state)
        assert func.final_value == pytest.approx(5.0)  # Final value updated

    def test_ignores_initial_velocity_acceleration(self):
        """Test that initial velocity/acceleration are ignored."""
        func = StepEndTimingFunction(
            initial_value=0.0, final_value=10.0, duration=10.0, initial_velocity=100.0, initial_acceleration=100.0
        )
        # State should be the same as if initial vel/accel were 0
        assert func.get_state(5.0) == pytest.approx(TimingState(value=0.0, velocity=0.0, acceleration=0.0))
        assert func(10.0) == pytest.approx(10.0)  # Jump value unaffected


class TestSmoothProp:
    def test_initialization(self):
        # Basic initialization
        prop = SmoothProp(5.0)
        assert prop.value == 5.0
        assert prop.duration == 0.0

        # With transition duration
        prop = SmoothProp(10.0, duration=100.0)
        assert prop.value == 10.0
        assert prop.duration == 100.0

    def test_immediate_transition(self):
        # With zero transition duration, changes should be immediate
        prop = SmoothProp(0.0, duration=0.0)
        prop.set(10.0)  # Set target value to 10.0
        assert prop.value == 10.0

        # Step should not affect the value
        prop.step(5.0)
        assert prop.value == 10.0

    def test_smooth_transition(self):
        # Create with transition duration
        prop = SmoothProp(0.0, duration=10.0)

        # Change value to trigger transition
        prop.set(1.0)  # Set target value to 1.0

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
        assert prop._interpolator is None

    def test_changing_target_mid_transition(self):
        prop = SmoothProp(0.0, duration=10.0)

        # Start transition to 1.0
        prop.set(1.0)  # Set target value to 1.0
        prop.step(2.0)

        # Value should have started moving toward 1.0
        assert 0.0 < prop.value < 1.0

        # Change target mid-transition
        prop.set(0.5)  # Change target to 0.5 (resets internal timer)

        # Complete the new transition (requires the full transition duration)
        prop.step(10.0)
        assert pytest.approx(prop.value, abs=1e-10) == 0.5

    def test_change_duration_mid_transition_shorter_than_current_time(self):
        # Test case where new duration < ctime < previous duration
        prop = SmoothProp(0.0, duration=10.0)

        # Start transition to 1.0
        prop.set(1.0)
        prop.step(7.0)  # Move 7 units into a 10-unit transition

        # Value should be moving toward 1.0
        original_value = prop.value
        assert 0.0 < original_value < 1.0

        # Change duration to 5.0 (which is < current time of 7.0)
        prop.set(value=None, duration=5.0)

        prop.step(2)
        new_value = prop.value
        assert original_value < new_value < 1.0
        prop.step(3)  # Complete the transition
        assert pytest.approx(prop.value, abs=1e-10) == 1.0

    def test_change_duration_mid_transition_shorter(self):
        # Test case where new ctime < duration < previous duration
        prop = SmoothProp(0.0, duration=10.0)

        # Start transition to 1.0
        prop.set(1.0)

        prop.step(3)

        # Value should be moving toward 1.0
        mid_value = prop.value
        assert 0.0 < mid_value < 1.0

        # Change duration to 7 (shorter than original, but we're still within new duration)
        prop.set(value=None, duration=7.0)

        assert pytest.approx(prop.value, abs=1e-10) == mid_value

        prop.step(7)

        assert pytest.approx(prop._ctime) == 7.0
        assert pytest.approx(prop.value, abs=1e-10) == 1.0

    def test_change_both_value_and_duration_mid_transition(self):
        # Test changing both value and duration simultaneously mid-transition
        prop = SmoothProp(0.0, duration=10.0)

        # Start transition to 1.0
        prop.set(1.0)
        prop.step(5.0)  # Move halfway through transition

        # Value should be moving toward 1.0
        mid_value = prop.value
        assert 0.0 < mid_value < 1.0

        # Change both value and duration mid-transition
        # Change direction (now aiming for -1.0) and extend duration to 15.0
        prop.set(value=-1.0, duration=15.0)

        # Verify value hasn't changed immediately after set() call
        assert pytest.approx(prop.value, abs=1e-10) == mid_value

        # Step partly through the new transition
        prop.step(7.5)  # Half of the new duration
        new_mid_value = prop.value

        # By this point the value should have started moving toward -1.0
        # It won't immediately decrease due to the minimum jerk physics,
        # but after half the duration it should definitely be lower than mid_value
        assert new_mid_value < mid_value

        # Complete the transition
        prop.step(7.5)  # This should complete the transition
        assert pytest.approx(prop.value, abs=1e-10) == -1.0

    def test_change_both_value_and_duration_shorter_transition(self):
        # Test changing to a shorter duration while also changing value
        prop = SmoothProp(0.0, duration=20.0)

        # Start transition to 1.0
        prop.set(1.0)
        prop.step(5.0)  # Move 5 units into transition

        initial_mid_value = prop.value
        assert 0.0 < initial_mid_value < 1.0

        # Change to a shorter duration (10.0) and new target value (2.0)
        prop.set(value=2.0, duration=10.0)

        # The transition should restart with new parameters
        assert prop._ctime == 0.0

        # Step halfway through new transition
        prop.step(5.0)
        new_mid_value = prop.value

        # Value should be moving toward 2.0 from where we were
        assert initial_mid_value < new_mid_value < 2.0

        # Complete the transition
        prop.step(5.0)
        assert pytest.approx(prop.value, abs=1e-10) == 2.0

    def test_zero_duration_initialization(self):
        """Test SmoothProp initialization with zero duration."""
        prop = SmoothProp(5.0, duration=0.0)
        assert prop.value == 5.0
        assert prop.duration == 0.0
        prop.set(10.0)  # Should be immediate
        assert prop.value == 10.0
        prop.step()
        assert prop.value == 10.0

    def test_zero_duration_set(self):
        """Test setting SmoothProp with zero duration."""
        prop = SmoothProp(0.0, duration=10.0)  # Start with non-zero duration
        prop.set(1.0)  # Start transition
        prop.step(5.0)
        assert 0.0 < prop.value < 1.0

        # Set with zero duration - should jump immediately
        prop.set(20.0, duration=0.0)
        assert prop.value == 20.0
        assert prop.duration == 0.0
        assert prop._interpolator is None
        prop.step()
        assert prop.value == 20.0  # Should remain 20.0

    def test_exact_completion(self):
        """Test that the value is exactly the final value upon completion."""
        duration = 10.0
        target_value = 1.0
        prop = SmoothProp(0.0, duration=duration)
        prop.set(target_value)

        # Step almost to the end
        prop.step(duration - 0.001)
        assert prop.value != target_value  # Should not be exactly there yet

        # Step exactly to the end
        prop.step(0.001)
        assert prop._ctime == pytest.approx(duration)
        # Due to the logic in step(), the interpolator might be cleared *after* this step,
        # but the value property should return the final_value correctly.
        assert prop.value == target_value  # Should be exactly the target value
        # Verify interpolator is cleared after accessing value or stepping again
        prop.step(0.001)  # Step slightly past
        assert prop._interpolator is None
        assert prop.value == target_value  # Should remain at target

    def test_smooth_transition_linear(self):
        """Test smooth transition using LinearTimingFunction."""
        prop = SmoothProp(0.0, duration=10.0, timing_function_cls=LinearTimingFunction)
        prop.set(1.0)
        prop.step(5.0)
        assert prop.value == pytest.approx(0.5)  # Exactly halfway
        prop.step(5.0)
        assert prop.value == pytest.approx(1.0)
        assert prop._interpolator is None

    def test_smooth_transition_step_end(self):
        """Test smooth transition using StepEndTimingFunction."""
        prop = SmoothProp(0.0, duration=10.0, timing_function_cls=StepEndTimingFunction)
        prop.set(1.0)
        prop.step(5.0)
        assert prop.value == pytest.approx(0.0)  # Stays at initial
        prop.step(4.999)
        assert prop.value == pytest.approx(0.0)
        prop.step(0.001)  # Step exactly to duration
        assert prop.value == pytest.approx(1.0)  # Jumps to final
        assert prop._interpolator is None
        prop.step(1.0)
        assert prop.value == pytest.approx(1.0)  # Stays at final

    def test_changing_target_mid_transition_linear(self):
        """Test changing target mid-transition with LinearTimingFunction."""
        prop = SmoothProp(0.0, duration=10.0, timing_function_cls=LinearTimingFunction)
        prop.set(1.0)
        prop.step(5.0)  # Value is now 0.5
        assert prop.value == pytest.approx(0.5)

        prop.set(0.0, duration=5.0)  # New target 0.0, duration 5.0
        assert prop.value == pytest.approx(0.5)  # Value doesn't change immediately
        assert prop._ctime == 0.0
        assert prop.duration == 5.0

        prop.step(2.5)  # Halfway through new transition
        # Should go from 0.5 to 0.0 over 5 steps. Halfway is 0.25
        assert prop.value == pytest.approx(0.25)

        prop.step(2.5)  # Complete transition
        assert prop.value == pytest.approx(0.0)
        assert prop._interpolator is None

    def test_changing_target_mid_transition_step_end(self):
        """Test changing target mid-transition with StepEndTimingFunction."""
        prop = SmoothProp(0.0, duration=10.0, timing_function_cls=StepEndTimingFunction)
        prop.set(1.0)
        prop.step(5.0)  # Value is still 0.0
        assert prop.value == pytest.approx(0.0)

        prop.set(2.0, duration=8.0)  # New target 2.0, duration 8.0
        assert prop.value == pytest.approx(0.0)  # Value doesn't change immediately
        assert prop._ctime == 0.0
        assert prop.duration == 8.0

        prop.step(7.999)  # Just before the jump
        assert prop.value == pytest.approx(0.0)

        prop.step(0.001)  # Exactly at the jump
        assert prop.value == pytest.approx(2.0)
        assert prop._interpolator is None

        prop.step(1.0)
        assert prop.value == pytest.approx(2.0)
