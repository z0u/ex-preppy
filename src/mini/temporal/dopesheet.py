from dataclasses import dataclass
import logging
import re
from typing import Literal, overload

import pandas as pd
from pandas.io.formats.style import Styler


log = logging.getLogger(__name__)


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
    phase_start: bool
    """Whether the phase is starting on this step."""
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
    - *: Other columns are interpreted as parameters to set

    https://en.m.wikipedia.org/wiki/Exposure_sheet
    """

    _df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self._df = resolve(df.copy())

    def __len__(self):
        """Get the number of steps in the dope sheet."""
        return self._df['STEP'].max()

    def __getitem__(self, step: int) -> Step:
        """
        Get the step details for the given step number.

        The sheet may not contain a keyframe for the given step. In that case, the
        current phase details will be returned without any keyed properties.
        """
        # Get the index of the nearest step that is less than or equal to `step`
        idx = self._df[self._df['STEP'] <= step].last_valid_index() or 0

        # Find the most recent phase before or at this step
        phase_idx = self._df[self._df['STEP'] <= step]['PHASE'].last_valid_index() or 0
        phase = self._df['PHASE'][phase_idx] or ''
        phase_start = self._df['STEP'][phase_idx] == step

        _t: int = self._df['STEP'][idx]
        if _t != step:
            # Not a keyframe; just return current phase details
            return Step(t=step, phase=phase, phase_start=False, actions=[], keyed_props=[])

        # Handle NaN values in the ACTION column
        action_value = self._df['ACTION'][idx]
        if pd.isna(action_value):
            actions = ['']
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

        return Step(t=step, phase=phase, phase_start=phase_start, actions=actions, keyed_props=keyed_props)

    @property
    def props(self) -> list[str]:
        """List of all properties in the dopesheet."""
        return [col for col in self._df.columns if col not in ['STEP', 'PHASE', 'ACTION']]

    @classmethod
    def from_csv(cls, path: str):
        """
        Load a dopesheet from a CSV file.

        The CSV file should have the following columns:
        - STEP: The step number (can be relative, e.g., +0.5)
        - PHASE: The phase of the curriculum (optional)
        - ACTION: The action to take (event to emit) (optional)
        - *: Other columns are interpreted as parameters to set

        Example:
            STEP,PHASE,ACTION,lr
            0,Basic,,0.0
            +0.5,,snapshot,0.01
            1000,,,0.001
        """
        df = pd.read_csv(path, dtype={'STEP': str, 'PHASE': str, 'ACTION': str})
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
