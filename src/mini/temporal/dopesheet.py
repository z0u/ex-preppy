import logging
import re
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

log = logging.getLogger(__name__)

RESERVED_COLS = ('STEP', 'PHASE', 'ACTION')
DEFAULT_SPACE = 'linear'
DEFAULT_INTERPOLATOR = 'minjerk'


@dataclass
class PropConfig:
    """Configuration for a single property column."""
    prop: str  # Base name, e.g., 'x'
    space: str = DEFAULT_SPACE
    interpolator_name: str = DEFAULT_INTERPOLATOR


@dataclass
class Key:
    prop: str
    """Name of the property this keyframe is for."""
    t: int
    """The frame number."""
    value: float
    """The value at this step."""
    next_t: int | None
    """Frame number of the next keyframe for this property."""
    next_value: float | None
    """Value at the next keyframe for this property."""

    @property
    def duration(self) -> int | None:
        """Duration of the transition starting at this key."""
        if self.next_t is None:
            return None
        return self.next_t - self.t


@dataclass
class Step:
    t: int
    phase: str
    """The current phase active *at* this step."""
    is_phase_start: bool
    """Whether this is the first step of the phase."""
    is_phase_end: bool
    """Whether this is the last step of the phase."""
    actions: list[str]
    """Actions listed at this step."""
    keyed_props: list[Key]
    """All properties that are keyed (have a non-NaN value) on this step."""


class Dopesheet:
    """
    A class to represent a dope sheet for parameter keyframes.

    ## Background
    A dope sheet (or exposure sheet) is a tool used in animation to organize and plan
    the timing of keyframes and actions. It typically helps animators visualize the
    sequence of events and manage the timing of actions effectively. It consists of a
    grid, where each row is a step in the animation, and each column represents a
    different property or action.

    ## Structure
    Dope sheets as defined by this class have the following columns:
    - STEP: The step/frame/epoch number
    - PHASE: The name of the phase of the curriculum (optional)
    - ACTION: The action to take (event to emit) (optional)
    - *: Other columns are interpreted as parameters to vary over time.

    https://en.m.wikipedia.org/wiki/Exposure_sheet
    """

    _df: pd.DataFrame
    _prop_configs: dict[str, PropConfig] = field(default_factory=dict)
    _phase_indices: np.ndarray

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Dopesheet with a DataFrame.

        Parses column headers for property configurations (e.g., 'x:log:minjerk')
        and resolves relative timesteps.

        See `from_csv`.
        """
        parsed_df, prop_configs = self._parse_header(df.copy())
        self._prop_configs = prop_configs
        self._df = resolve(parsed_df)
        self._phase_indices = self._df['PHASE'].dropna().index.to_numpy()

    @staticmethod
    def _parse_header(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, PropConfig]]:
        """Parse property configurations from DataFrame headers."""
        prop_configs = {}
        rename_map = {}
        pattern = re.compile(r'^(?P<prop>[^:]+)(?::(?P<space>[^:]+))?(?::(?P<interpolator>[^:]+))?$')

        for col in df.columns:
            if col in RESERVED_COLS:
                continue

            match = pattern.match(col)
            if match:
                parts = match.groupdict()
                prop_name = parts['prop']
                config = PropConfig(
                    prop=prop_name,
                    space=parts.get('space') or DEFAULT_SPACE,
                    interpolator_name=parts.get('interpolator') or DEFAULT_INTERPOLATOR,
                )
                prop_configs[prop_name] = config
                if col != prop_name:
                    rename_map[col] = prop_name
            else:
                # Assume default config if no pattern matches but not reserved
                prop_configs[col] = PropConfig(prop=col)
                log.warning(f"Column '{col}' doesn't match 'prop:space:interpolator' format. Assuming defaults.")

        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure all identified props exist in the final config dict
        final_props = [c for c in df.columns if c not in RESERVED_COLS]
        for prop in final_props:
            if prop not in prop_configs:
                prop_configs[prop] = PropConfig(prop=prop)  # Add with defaults if missed

        return df, prop_configs

    def __len__(self):
        """Get the number of steps in the dope sheet."""
        return self._df['STEP'].max() + 1

    def __getitem__(self, step: int) -> Step:
        """
        Get the step details for the given step number.

        The sheet may not contain a keyframe for the given step. In that case, the
        current phase details will be returned without any keyed properties.
        """
        steps_col = self._df['STEP']

        # Use binary search (searchsorted) to find the index of the latest step <= input step
        # 'right' side gives the insertion point index `i`. The index we want is `i-1`.
        insertion_point = steps_col.searchsorted(step, side='right')
        idx = max(0, insertion_point - 1)

        # --- OPTIMIZED PHASE LOOKUP ---
        # Find the most recent phase index using binary search on precomputed indices
        phase_insertion_point = np.searchsorted(self._phase_indices, idx, side='right')

        if phase_insertion_point == 0:
            # No phase definition found at or before idx in our precomputed list.
            # Mimic original fallback to index 0. A clearer default might be better,
            # but this maintains existing behavior.
            phase_idx = 0
        else:
            # Get the index from our precomputed list (index before insertion point)
            phase_idx = self._phase_indices[phase_insertion_point - 1]
        # --- END OPTIMIZED PHASE LOOKUP ---

        # Now get the phase value and check start using the efficiently found phase_idx
        phase = self._df['PHASE'][phase_idx] or ''
        phase_start = steps_col[phase_idx] == step  # Check if the step where phase was defined IS the current step

        # --- Calculate Phase End ---
        if phase_insertion_point < len(self._phase_indices):
            # There is a next phase defined. Find its start step.
            next_phase_df_idx = self._phase_indices[phase_insertion_point]
            next_phase_start_step = steps_col[next_phase_df_idx]
            # The current phase ends one step before the next one starts
            current_phase_end_step = next_phase_start_step - 1
        else:
            # This is the last phase, it ends at the last step of the dopesheet
            current_phase_end_step = steps_col.max()  # Get the maximum step value

        is_phase_end = step == current_phase_end_step
        # --- End Calculate Phase End ---

        _t: int = steps_col[idx]
        if _t != step:
            # Not a keyframe; just return current phase details
            return Step(
                t=step,
                phase=phase,
                is_phase_start=False,
                is_phase_end=is_phase_end,
                actions=[],
                keyed_props=[],
            )

        # Handle NaN values in the ACTION column
        action_value = self._df['ACTION'][idx]
        if pd.isna(action_value) or action_value == '':
            actions = []
        else:
            actions = str(action_value).split(',')

        keyed_props = []
        for prop in self.props:
            series = self._df[prop]
            value = series[idx]
            if pd.isna(value):
                continue
            next_idx: int | None = series[series.index > idx].first_valid_index()  # type: ignore
            k = Key(
                prop=prop,
                t=step,
                value=value,
                next_t=self._df['STEP'][next_idx] if next_idx is not None else None,
                next_value=series[next_idx] if next_idx is not None else None,
            )
            keyed_props.append(k)

        return Step(
            t=step,
            phase=phase,
            is_phase_start=phase_start,
            is_phase_end=is_phase_end,
            actions=actions,
            keyed_props=keyed_props,
        )

    @property
    def props(self) -> list[str]:
        """List of all base property names in the dopesheet."""
        return [col for col in self._df.columns if col not in RESERVED_COLS]

    def get_prop_config(self, prop_name: str) -> PropConfig:
        """Get the parsed configuration for a specific property."""
        # Return the stored config, defaulting to a default PropConfig if somehow missed
        return self._prop_configs.get(prop_name, PropConfig(prop=prop_name))

    @property
    def phases(self) -> set[str]:
        """Return a set of unique phase names defined in the dopesheet."""
        return set(self._df['PHASE'].dropna().unique())

    def get_initial_values(self) -> dict[str, float]:
        """
        Get the initial value for each property in the dopesheet.

        For each property, finds the first non-NaN value in the timeline.
        If a property has no values, it will not be included in the result.

        Returns:
            A dictionary mapping property names to their initial values.
        """
        initial_values = {}
        for prop in self.props:
            # Find the first non-NaN value for this property
            series = self._df[prop]
            first_valid_idx = series.first_valid_index()
            if first_valid_idx is not None:
                initial_values[prop] = float(series[first_valid_idx])
        return initial_values

    @classmethod
    def from_csv(cls, path: Path | str | BytesIO | StringIO) -> 'Dopesheet':
        """
        Load a dopesheet from a CSV file.

        The CSV file should have the following columns:
        - STEP: The step number (can be relative, e.g., +0.5)
        - PHASE: The phase of the curriculum (optional)
        - ACTION: The action to take (event to emit) (optional)
        - *: Other columns are interpreted as parameters to set.
             These can be in the format 'prop:space:interpolator' (e.g., 'lr:log:minjerk')
             or just 'prop' (e.g., 'momentum'), which implies defaults ('linear', 'minjerk').

        Example:
            STEP,PHASE,ACTION,lr:log:minjerk,momentum
            0,Basic,,0.01,0.9
            +0.5,,snapshot,0.005,
            1000,,,0.001,0.99
        """
        # Assuming header=0 works correctly even with complex names
        df = pd.read_csv(path, dtype={'STEP': str, 'PHASE': str, 'ACTION': str}, header=0)
        return cls(df)

    @overload
    def as_df(self, *, styled: Literal[True]) -> Styler: ...
    @overload
    def as_df(self, *, styled: Literal[False] = False) -> pd.DataFrame: ...

    def as_df(self, *, styled=False) -> pd.DataFrame | Styler:
        """Convert the dopesheet to a pandas DataFrame."""
        df = self._df.copy()
        if styled:
            df = style_dopesheet(df)
        return df

    def to_markdown(self) -> str:
        mdtable = self._df.to_markdown(index=False, tablefmt='pipe')
        # Run twice to account for overlapping matches
        mdtable = re.sub(r'(\|\s*)nan(\s*\|)', r'\1   \2', mdtable, flags=re.IGNORECASE)
        mdtable = re.sub(r'(\|\s*)nan(\s*\|)', r'\1   \2', mdtable, flags=re.IGNORECASE)
        return mdtable


def style_dopesheet(df: pd.DataFrame) -> Styler:
    import pandas as pd

    decimal_places: dict[str, int] = {}
    non_numeric_cols: list[int] = []

    for i, col in enumerate(df.columns):
        # Check if the column's dtype is a subtype of number
        if pd.api.types.is_numeric_dtype(df[col]):
            # Drop NaNs for precision calculation
            col_no_na = df[col].dropna()

            if col_no_na.empty:
                # Default to 2 if all are NaN
                precision = 2
            else:
                # Check if all numbers are essentially integers
                is_integer = (col_no_na == col_no_na.round(0)).all()
                if is_integer:
                    precision = 0
                else:
                    # Calculate max decimal places needed
                    precision = col_no_na.astype(str).str.split('.', expand=True)[1].str.len().max()
                    # Handle cases where a column might have only integers after dropping NaNs
                    precision = int(precision) if pd.notna(precision) else 0
            decimal_places[col] = min(precision, 6)
        elif col == 'STEP':
            pass
        else:
            non_numeric_cols.append(i)

    log.info(f'Calculated decimal places: {decimal_places}')
    log.info(f'Non-numeric columns: {non_numeric_cols}')

    # Let's refine the display style based on this
    style = df.style.set_table_styles(
        [
            {'selector': 'td,th', 'props': 'white-space: nowrap'},
            # Left-align non-numeric columns
            *[{'selector': f'.col{i}', 'props': 'text-align: left'} for i in non_numeric_cols],
        ]
    ).format(na_rep='')
    for i, precision in decimal_places.items():
        style = style.format(na_rep='', precision=precision, subset=[i])

    return style


def resolve(df: pd.DataFrame) -> pd.DataFrame:
    df['STEP'] = resolve_timesteps(df['STEP'])
    df = df.sort_values(by='STEP', ignore_index=True).reset_index(drop=True)
    return df


def resolve_timesteps(steps: pd.Series) -> pd.Series:
    # Ensure steps exists and STEP is string
    steps = steps.astype(str)

    # Initialize the new column with NaNs
    _steps = pd.Series(pd.NA, index=steps.index)

    # --- 1. Identify Anchors ---
    anchor_indices = steps.index[~steps.str.startswith('+')]
    anchor_steps = {}
    for idx in anchor_indices:
        try:
            step_val = int(steps.loc[idx])  # type: ignore
            anchor_steps[idx] = step_val
            # Assign anchor steps directly to the new column
            _steps.loc[idx] = step_val
        except ValueError:
            # Handle cases where a non-relative step isn't an integer (shouldn't happen here)
            log.warning(f"Warning: Non-relative step '{steps.loc[idx]}' at index {idx} is not an integer.")

    # --- 2. Iterate and Interpolate ---
    relative_indices = steps.index[steps.str.startswith('+')]

    for idx in relative_indices:
        step_str = steps.loc[idx]
        try:
            fraction = float(step_str[1:])  # type: ignore
        except ValueError:
            log.warning(f"Warning: Could not parse fraction from '{step_str}' at index {idx}.")
            continue

        # Find preceding anchor index (largest anchor index < current index)
        prev_anchor_idx = max((a_idx for a_idx in anchor_indices if a_idx < idx), default=-1)

        # Find succeeding anchor index (smallest anchor index > current index)
        next_anchor_idx = min((a_idx for a_idx in anchor_indices if a_idx > idx), default=-1)

        # Check if we found valid anchors
        if prev_anchor_idx == -1 or next_anchor_idx == -1:
            log.warning(f'Warning: Could not find bracketing anchor steps for relative step at index {idx}.')
            continue

        # Get anchor step values
        prev_step = anchor_steps[prev_anchor_idx]
        next_step = anchor_steps[next_anchor_idx]

        # Perform linear interpolation and round
        interpolated_step = prev_step + fraction * (next_step - prev_step)
        _steps.loc[idx] = round(interpolated_step)

    # Convert the result to a nullable integer type
    return _steps.astype('Int64')
