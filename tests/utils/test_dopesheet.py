import pandas as pd
import pytest
from pathlib import Path

from utils.dopesheet import Dopesheet, Key, Step


@pytest.fixture
def fixture_path() -> Path:
    """Return the path to the fixture file."""
    return Path(__file__).parent / 'fixtures' / 'dopesheet.csv'


@pytest.fixture
def dopesheet(fixture_path) -> Dopesheet:
    """Return a Dopesheet loaded from the fixture file."""
    return Dopesheet.from_csv(str(fixture_path))


class TestDopesheet:
    def test_init_from_csv(self, fixture_path):
        """Test that we can initialize a Dopesheet from a CSV file."""
        ds = Dopesheet.from_csv(str(fixture_path))
        assert isinstance(ds, Dopesheet)

        # Check that the dataframe was loaded correctly
        df = ds._df
        assert list(df.columns) == ['STEP', 'PHASE', 'ACTION', 'x', 'y', 'z']
        assert len(df) == 4  # Four rows in the fixture

        # Check that steps were resolved correctly
        assert list(df['STEP']) == [0, 4, 10, 11]  # +0.4 should become 4

    def test_len(self, dopesheet):
        """Test the __len__ method."""
        # Length should be the max step value
        assert len(dopesheet) == 11

    def test_props(self, dopesheet):
        """Test the props property."""
        assert dopesheet.props == ['x', 'y', 'z']

    def test_get_keyframe_steps(self, dopesheet):
        """Test __getitem__ for steps that are keyframes."""
        # Step 0
        step0 = dopesheet[0]
        assert isinstance(step0, Step)
        assert step0.t == 0
        assert step0.phase == 'One'
        assert step0.phase_start is True
        assert step0.actions == ['']  # Empty action becomes [""]

        # Check keyed properties for step 0
        assert len(step0.keyed_props) == 2  # x and z are set
        x_key = next(k for k in step0.keyed_props if k.prop == 'x')
        z_key = next(k for k in step0.keyed_props if k.prop == 'z')

        assert x_key.value == 0.01
        assert x_key.next_t == 10  # Fixed: Next x value is at step 10, not 4
        assert x_key.next_value == 0.001  # Fixed: Next x value is 0.001, not 0.0
        assert x_key.duration == 10  # Fixed: Duration is 10 steps, not 4

        assert z_key.value == 1
        assert z_key.next_t == 4
        assert z_key.next_value == 2
        assert z_key.duration == 4

        # Step 4 (the resolved +0.4)
        step4 = dopesheet[4]
        assert step4.t == 4
        assert step4.phase == 'One'  # Phase is carried forward
        assert step4.phase_start is True
        assert step4.actions == ['foo']

        # Check keyed properties for step 4
        assert len(step4.keyed_props) == 2  # y and z are set
        y_key = next(k for k in step4.keyed_props if k.prop == 'y')
        z_key = next(k for k in step4.keyed_props if k.prop == 'z')

        assert y_key.value == 0.8
        assert y_key.next_t == 11
        assert y_key.next_value == 0
        assert y_key.duration == 7

        assert z_key.value == 2
        assert z_key.next_t == 10
        assert z_key.next_value == 3
        assert z_key.duration == 6

        # Step 10
        step10 = dopesheet[10]
        assert step10.t == 10
        assert step10.phase == 'Two'
        assert step10.phase_start is True
        assert step10.actions == ['']

        # Check keyed properties for step 10
        assert len(step10.keyed_props) == 2  # x and z are set
        x_key = next(k for k in step10.keyed_props if k.prop == 'x')
        z_key = next(k for k in step10.keyed_props if k.prop == 'z')

        assert x_key.value == 0.001
        assert x_key.next_t is None
        assert x_key.next_value is None
        assert x_key.duration is None

        assert z_key.value == 3
        assert z_key.next_t == 11
        assert z_key.next_value == 4
        assert z_key.duration == 1

        # Step 11
        step11 = dopesheet[11]
        assert step11.t == 11
        assert step11.phase == 'Fin'
        assert step11.phase_start is True
        assert step11.actions == ['']

        # Check keyed properties for step 11
        assert len(step11.keyed_props) == 2  # y and z are set
        y_key = next(k for k in step11.keyed_props if k.prop == 'y')
        z_key = next(k for k in step11.keyed_props if k.prop == 'z')

        assert y_key.value == 0.0
        assert y_key.next_t is None
        assert y_key.next_value is None
        assert y_key.duration is None

        assert z_key.value == 4
        assert z_key.next_t is None
        assert z_key.next_value is None
        assert z_key.duration is None

    def test_get_non_keyframe_steps(self, dopesheet):
        """Test __getitem__ for steps that are not keyframes."""
        # Step 2 (between 0 and 4)
        step2 = dopesheet[2]
        assert step2.t == 2
        assert step2.phase == 'One'  # Phase from previous keyframe
        assert step2.phase_start is False  # Not the start of the phase
        assert step2.actions == []  # No actions on non-keyframe steps
        assert step2.keyed_props == []  # No keyed properties on non-keyframe steps

        # Step 5 (between 4 and 10)
        step5 = dopesheet[5]
        assert step5.t == 5
        assert step5.phase == 'One'
        assert step5.phase_start is False
        assert step5.actions == []
        assert step5.keyed_props == []

        # Step 10.5 (not a valid step, but should return closest previous keyframe's phase)
        step10_5 = dopesheet[10.5]
        assert step10_5.t == 10.5
        assert step10_5.phase == 'Two'
        assert step10_5.phase_start is False
        assert step10_5.actions == []
        assert step10_5.keyed_props == []
