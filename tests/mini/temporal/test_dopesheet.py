from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from mini.temporal.dopesheet import Dopesheet, resolve_timesteps
from mini.temporal.model import Keyframe, PropConfig, Frame


@pytest.fixture
def fixture_path() -> Path:
    """Return the path to the fixture file."""
    return Path(__file__).parent / 'fixtures' / 'dopesheet.csv'


@pytest.fixture
def dopesheet(fixture_path) -> Dopesheet:
    """Return a Dopesheet loaded from the fixture file."""
    return Dopesheet.from_csv(str(fixture_path))


@pytest.mark.parametrize(
    'input_steps, expected_steps',
    [
        # Basic absolute
        (['0', '10', '20'], [0, 10, 20]),
        # Basic fractional (+0.x)
        (['0', '+0.5', '10'], [0, 5, 10]),
        # Positive integer offset (+N)
        (['0', '+5', '10'], [0, 5, 10]),
        (['10', '+2', '20'], [10, 12, 20]),
        # Negative integer offset (-N)
        (['0', '-3', '10'], [0, 7, 10]),
        (['10', '-1', '20'], [10, 19, 20]),
        # Negative fractional offset (-0.x)
        (['0', '-0.5', '10'], [0, 5, 10]),  # 10 - 0.5 * (10 - 0) = 5
        (['10', '-0.2', '20'], [10, 18, 20]),  # 20 - 0.2 * (20 - 10) = 18
        # Mixed types
        (['0', '+2', '+0.5', '-1', '10'], [0, 2, 5, 9, 10]),  # +2 -> 0+2=2; +0.5 -> 0+0.5*(10-0)=5; -1 -> 10-1=9
        # Edge case: Clamping negative results
        (['0', '-5', '3'], [0, 0, 3]),  # 3 - 5 = -2 -> 0
        (['5', '-10', '8'], [5, 0, 8]),  # 8 - 10 = -2 -> 0
        # Edge case: +N without preceding anchor (should be NaN/NA)
        (['+5', '10'], [pd.NA, 10]),
        # Edge case: -N without succeeding anchor (should be NaN/NA)
        (['0', '-5'], [0, pd.NA]),
        # Edge case: +0.x without bracketing anchors (should be NaN/NA)
        (['+0.5', '10'], [pd.NA, 10]),
        (['0', '+0.5'], [0, pd.NA]),
        # Edge case: -0.x without bracketing anchors (should be NaN/NA)
        (['-0.5', '10'], [pd.NA, 10]),
        (['0', '-0.5'], [0, pd.NA]),
        # Edge case: Invalid format (should be NaN/NA)
        (['0', '+abc', '10'], [0, pd.NA, 10]),
        (['0', '-xyz', '10'], [0, pd.NA, 10]),
        (['0', '-', '10'], [0, pd.NA, 10]),
        # Edge case: Negative step at start interpreted as relative (-N)
        (['-5', '10'], [5, 10]),  # Resolves to 10 - 5 = 5
        # Edge case: Fractional step outside (0, 1) (should be NaN/NA)
        (['0', '+1.5', '10'], [0, pd.NA, 10]),
        (['0', '-0.5', '10'], [0, 5, 10]),  # This one is valid now
        (['0', '-1.5', '10'], [0, pd.NA, 10]),
    ],
)
def test_resolve_timesteps_relative(input_steps, expected_steps):
    """Test resolve_timesteps with various relative step formats."""
    input_series = pd.Series(input_steps, dtype=str)
    # Use Int64 for expected to handle pd.NA
    expected_series = pd.Series(expected_steps, dtype='Int64')
    resolved_series = resolve_timesteps(input_series)
    assert_series_equal(resolved_series, expected_series, check_dtype=True)


class TestDopesheet:
    def test_init_from_csv(self, fixture_path: Path):
        """Test that we can initialize a Dopesheet from a CSV file."""
        ds = Dopesheet.from_csv(str(fixture_path))
        assert isinstance(ds, Dopesheet)

        # Check that the dataframe was loaded correctly
        df = ds._df
        assert list(df.columns) == ['STEP', 'PHASE', 'ACTION', 'x', 'y', 'z']
        assert len(df) == 5  # Five rows in the fixture

        # Check that steps were resolved correctly
        assert list(df['STEP']) == [0, 4, 10, 11, 12]  # +0.4 should resolve to 4

    def test_len(self, dopesheet: Dopesheet):
        """Test the __len__ method."""
        # Length should be the max step value plus one because the first step is numbered 0
        assert len(dopesheet) == 13

    def test_phases(self, dopesheet: Dopesheet):
        """Test the phases property."""
        assert dopesheet.phases == {'One', 'Two', 'Fin'}

    def test_props(self, dopesheet: Dopesheet):
        """Test the props property."""
        assert dopesheet.props == ['x', 'y', 'z']

    def test_get_keyframe_steps(self, dopesheet: Dopesheet):
        """Test __getitem__ for steps that are keyframes."""
        # Step 0
        assert dopesheet[0] == Frame(
            t=0,
            phase='One',
            is_phase_start=True,
            is_phase_end=False,
            actions=[],
            keyed_props=[
                Keyframe(prop='x', t=0, value=0.01, next_t=10, next_value=0.001),
                Keyframe(prop='z', t=0, value=1, next_t=4, next_value=2),
            ],
        )

        # Step 4 (the resolved +0.4)
        assert dopesheet[4] == Frame(
            t=4,
            phase='One',  # Phase is carried forward
            is_phase_start=False,
            is_phase_end=False,
            actions=['foo'],
            keyed_props=[
                Keyframe(prop='y', t=4, value=0.8, next_t=11, next_value=0),
                Keyframe(prop='z', t=4, value=2, next_t=10, next_value=3),
            ],
        )

        # Step 10
        assert dopesheet[10] == Frame(
            t=10,
            phase='Two',
            is_phase_start=True,
            is_phase_end=True,  # End of phase "Two" (since phase "Fin" starts at step 11)
            actions=[],
            keyed_props=[
                Keyframe(prop='x', t=10, value=0.001, next_t=None, next_value=None),
                Keyframe(prop='z', t=10, value=3, next_t=11, next_value=4),
            ],
        )

        # Step 11
        assert dopesheet[11] == Frame(
            t=11,
            phase='Fin',
            is_phase_start=True,
            is_phase_end=False,
            actions=[],
            keyed_props=[
                Keyframe(prop='y', t=11, value=0.0, next_t=None, next_value=None),
                Keyframe(prop='z', t=11, value=4, next_t=None, next_value=None),
            ],
        )

        # Step 12
        assert dopesheet[12] == Frame(
            t=12,
            phase='Fin',
            is_phase_start=False,
            is_phase_end=True,
            actions=[],
            keyed_props=[],
        )

    def test_get_non_keyframe_steps(self, dopesheet: Dopesheet):
        """Test __getitem__ for steps that are not keyframes."""
        # Step 2 (between 0 and 4)
        assert dopesheet[2] == Frame(
            t=2,
            phase='One',
            is_phase_start=False,
            is_phase_end=False,
            actions=[],
            keyed_props=[],
        )

        # Step 5 (between 4 and 10)
        assert dopesheet[5] == Frame(
            t=5,
            phase='One',
            is_phase_start=False,
            is_phase_end=False,
            actions=[],
            keyed_props=[],
        )

    def test_get_initial_values(self, dopesheet: Dopesheet):
        """Test the get_initial_values method."""
        initial_values = dopesheet.get_initial_values()

        # Based on our fixture CSV:
        # - 'x' first appears at step 0 with value 0.01
        # - 'y' first appears at step 4 with value 0.8
        # - 'z' first appears at step 0 with value 1
        assert initial_values == {'x': 0.01, 'y': 0.8, 'z': 1}

    def test_prop_config_parsing(self, dopesheet: Dopesheet):
        """
        Test that property configurations are correctly parsed from column headers.

        Tests the new position-based syntax for header formats:
        - 'prop' (e.g., 'y') - Uses defaults for both space and interpolator
        - 'prop:space' (e.g., 'x:log') - Customizes space, uses default interpolator
        - 'prop::interpolator' (e.g., 'z::step-end') - Uses default space, customizes interpolator
        """
        # Check that the configs were stored correctly
        assert dopesheet._prop_configs.keys() == {'x', 'y', 'z'}

        # Check x config (x:log - space only)
        x_config = dopesheet.get_prop_config('x')
        assert isinstance(x_config, PropConfig)
        assert x_config.prop == 'x'
        assert x_config.space == 'log'
        assert x_config.timing_fn == 'minjerk'  # Default

        # Check y config (no configs - all defaults)
        y_config = dopesheet.get_prop_config('y')
        assert y_config.prop == 'y'
        assert y_config.space == 'linear'  # Default
        assert y_config.timing_fn == 'minjerk'  # Default

        # Check z config (z::step-end - interpolator only)
        z_config = dopesheet.get_prop_config('z')
        assert z_config.prop == 'z'
        assert z_config.space == 'linear'  # Default
        assert z_config.timing_fn == 'step-end'
