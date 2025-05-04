from pathlib import Path
from unittest.mock import ANY

import pytest
from pytest import approx

from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import State, Timeline


@pytest.fixture
def dopesheet_path():
    """Returns the path to the test fixture dopesheet."""
    fixture_path = Path(__file__).parent / 'fixtures' / 'dopesheet.csv'
    return str(fixture_path)


@pytest.fixture
def dopesheet(dopesheet_path):
    """Returns a Dopesheet instance loaded from the fixture."""
    return Dopesheet.from_csv(dopesheet_path)


@pytest.fixture
def timeline(dopesheet):
    """Returns a Timeline instance with the test dopesheet."""
    return Timeline(dopesheet)


def test_timeline_initialization(dopesheet):
    """Test that the timeline initializes correctly with a dopesheet."""
    # Create a timeline
    timeline = Timeline(dopesheet)

    # Verify initial properties
    assert timeline._step == 0
    assert timeline.props.keys() == {'x', 'y', 'z'}

    # Verify initial values match the dopesheet's initial values
    initial_values = dopesheet.get_initial_values()
    assert timeline.props['x'].value == approx(initial_values['x'])
    assert timeline.props['y'].value == approx(initial_values['y'])
    assert timeline.props['z'].value == approx(initial_values['z'])


def test_timeline_step_progression(timeline):
    """Test that the timeline steps forward correctly."""
    # Initial state
    assert timeline._step == 0

    # Step forward
    state = timeline.step()
    assert timeline._step == 1
    assert state == State(
        step=1,
        phase='One',
        actions=[],
        props={'x': ANY, 'y': ANY, 'z': ANY},
        is_phase_start=False,
        is_phase_end=False,
    )


def test_timeline_property_transitions(timeline):
    """Test that property values transition appropriately between keyframes."""
    # Step to 1 - values should start moving towards the next keyframe
    state = timeline.step()
    assert timeline._step == 1

    # x should be transitioning with minjerk (log space)
    assert state.props['x'] != approx(0.01)

    # z should stay at 1.0 until we reach step 4, since it uses step-end
    assert state.props['z'] == approx(1.0)

    # Step to 4 (where y and z have keyframes)
    for _ in range(3):
        state = timeline.step()

    assert timeline._step == 4
    assert approx(state.props) == {'x': ANY, 'y': 0.8, 'z': 2.0}

    # Now step to 10 (where x and z have keyframes)
    for _ in range(6):
        state = timeline.step()

    assert timeline._step == 10
    assert approx(state.props) == {'x': 0.001, 'y': ANY, 'z': 3.0}

    # Final step to 11
    state = timeline.step()
    assert timeline._step == 11
    assert approx(state.props) == {'x': ANY, 'y': 0.0, 'z': 4.0}


def test_timeline_phase_and_actions(timeline):
    """Test that the timeline reports phases and actions correctly."""
    # Step 0 is phase "One" with no actions
    state = timeline.state
    assert state.phase == 'One'
    assert state.actions == []

    # Step to 4, which has an action "foo"
    for _ in range(4):
        state = timeline.step()

    assert state.step == 4
    assert state.phase == 'One'
    assert state.actions == ['foo']

    # Step to 10, which begins phase "Two"
    for _ in range(6):
        state = timeline.step()

    assert state.step == 10
    assert state.phase == 'Two'
    assert state.actions == []

    # Step to 11, which begins phase "Fin"
    state = timeline.step()
    assert state.step == 11
    assert state.phase == 'Fin'
    assert state.actions == []


def test_timeline_phase_transitions(timeline):
    """Test that phase transitions are correctly identified with is_phase_start and is_phase_end flags."""
    # Define expected phases structure based on our fixture:
    # - "One": steps 0-9
    # - "Two": step 10
    # - "Fin": step 11

    # Check initial state (step 0)
    state = timeline.state
    assert state == State(
        step=0,
        phase='One',
        actions=ANY,
        props=ANY,
        is_phase_start=True,
        is_phase_end=False,
    )

    # Step to 1 (middle of phase "One")
    state = timeline.step()
    assert state == State(
        step=1,
        phase='One',
        actions=ANY,
        props=ANY,
        is_phase_start=False,
        is_phase_end=False,
    )

    # Step to 9 (end of phase "One")
    for _ in range(8):  # Already at step 1, need 8 more steps to get to 9
        timeline.step()
    state = timeline.state
    assert state == State(
        step=9,
        phase='One',
        actions=ANY,
        props=ANY,
        is_phase_start=False,
        is_phase_end=True,
    )

    # Step to 10 (both start and end of phase "Two")
    state = timeline.step()
    assert state == State(
        step=10,
        phase='Two',
        actions=ANY,
        props=ANY,
        is_phase_start=True,
        is_phase_end=True,
    )

    # Step to 11 (start of phase "Fin" - final step)
    state = timeline.step()
    assert state == State(
        step=11,
        phase='Fin',
        actions=ANY,
        props=ANY,
        is_phase_start=True,
        is_phase_end=False,
    )

    # Step to 12 (start and end of phase "Fin" - final step)
    state = timeline.step()
    assert state == State(
        step=12,
        phase='Fin',
        actions=ANY,
        props=ANY,
        is_phase_start=False,
        is_phase_end=True,
    )


def test_interpolation_functions(timeline):
    """Test that different interpolation functions are correctly applied to properties."""
    # Our test fixture has:
    # - x: log:minjerk
    # - y: linear:minjerk (default)
    # - z: linear:step-end

    # Step to 2 (halfway between 0 and 4)
    timeline.step()
    timeline.step()
    state = timeline.state
    assert state.step == 2

    # Check that z uses step-end behavior (should remain at 1.0 until we reach step 4)
    # Since we're using StepEndTimingFunction, it should stay at exactly 1.0
    assert state.props['z'] == approx(1.0)

    # x should be changing (using minjerk in log space)
    assert state.props['x'] != approx(0.01)

    # y doesn't have a key at step 0, so it won't start changing until step 4
    # It should still be at its initial value
    assert state.props['y'] == approx(0.8)

    # Step to 5 (one step after z's first keyframe)
    timeline.step()
    timeline.step()
    timeline.step()
    state = timeline.state
    assert state.step == 5

    # z should now be at 2.0 because of step-end
    assert state.props['z'] == approx(2.0)

    # Step to 6 - y should now be changing
    timeline.step()
    state = timeline.state
    assert state.props['y'] != approx(0.8)  # Now it should be changing

    # Step to 8 (halfway to step 10)
    timeline.step()
    timeline.step()
    state = timeline.state
    assert state.step == 8

    # z should still be at 2.0 because of step-end (won't change until step 10)
    assert state.props['z'] == approx(2.0)
