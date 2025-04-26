from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mini.temporal.dopesheet import RESERVED_COLS
from mini.temporal.timeline import Timeline


def realize_timeline(timeline: Timeline) -> pd.DataFrame:
    # Collect data by stepping through the timeline
    history = []
    max_steps = len(timeline.dopesheet)
    for _ in range(max_steps + 1):
        state = timeline.state
        history.append({'STEP': state.step, 'PHASE': state.phase, 'ACTION': state.actions, **state.props})
        if state.step < max_steps:
            timeline.step()

    return pd.DataFrame(history)


@dataclass
class ParamGroup:
    """A group of parameters to be plotted together."""

    name: str
    params: list[str]
    height_ratio: float = 1.0
    """Height ratio for the subplot."""


def group_properties_by_scale(df: pd.DataFrame) -> tuple[ParamGroup, ParamGroup]:
    """Group properties by their scale for better visualization."""
    # Calculate statistics for each property
    prop_stats = {
        prop: {'max': df[prop].max(), 'median': df[prop].median(), 'min': df[prop].min()}  #
        for prop in df.columns
    }

    # Calculate median of medians as a representative scale
    median_of_medians = np.median([stats['median'] for stats in prop_stats.values()])

    # Group properties - those with much smaller medians go to group 2
    # Handle cases where median_of_medians might be zero or very small
    threshold = median_of_medians * 0.2 if median_of_medians > 1e-9 else 1e-9
    group1 = [prop for prop in df.columns if prop_stats[prop]['max'] >= threshold]
    group2 = [prop for prop in df.columns if prop_stats[prop]['max'] < threshold]

    return ParamGroup(name='', params=group1, height_ratio=2.0), ParamGroup(name='', params=group2, height_ratio=1.0)


def plot_timeline(history_df: pd.DataFrame, keyframes_df: pd.DataFrame, groups: Sequence[ParamGroup] | None = None):  # noqa: C901
    if groups is None:
        cols = [col for col in history_df.columns if col not in RESERVED_COLS]
        groups = [ParamGroup(name='', params=cols, height_ratio=1.0)]
    height_ratios = [g.height_ratio for g in groups]
    figsize = (15, 3.5 * len(groups))

    # Create figure and axes
    plt.style.use('dark_background')
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios},
        squeeze=False,  # Always return a 2D array
    )
    fig.set_facecolor('#333')

    # Plot each group on its corresponding axis
    for group, ax in zip(groups, axes.flatten(), strict=True):
        for prop in group.params:
            # Ensure the property exists in the history dataframe before plotting
            if prop in history_df.columns:
                ax.set_facecolor('#222')
                (line,) = ax.plot(history_df['STEP'], history_df[prop], label=f'{prop}')

                # Add markers for keyframes if the property exists in keyframes
                if prop in keyframes_df.columns:
                    prop_keyframes = keyframes_df.dropna(subset=[prop])
                    if not prop_keyframes.empty:
                        ax.scatter(
                            prop_keyframes['STEP'],
                            prop_keyframes[prop],
                            marker='o',
                            facecolor='black',
                            s=25,
                            zorder=5,
                            color=line.get_color(),
                        )
            else:
                print(f"Warning: Property '{prop}' specified in group '{group.name}' not found in history_df.")

    # Indicate phase changes and add labels (only on the top plot)
    phase_changes = keyframes_df.dropna(subset=['PHASE'])
    last_phase = None
    phase_boundaries = []
    for _, row in phase_changes.iterrows():
        if row['PHASE'] != last_phase:
            step = row['STEP']
            # Draw vertical line on all axes
            for ax in axes.flatten():
                ax.axvline(step, color='grey', alpha=0.2)
            phase_boundaries.append({'STEP': step, 'PHASE': row['PHASE']})
            last_phase = row['PHASE']

    # Add phase labels only to the top plot (axes[0, 0])
    top_ax = axes[0, 0]
    for i, pb in enumerate(phase_boundaries):
        mid_point = (
            (phase_boundaries[i + 1]['STEP'] + pb['STEP']) / 2
            if i + 1 < len(phase_boundaries)
            else (len(history_df) + pb['STEP']) / 2
        )
        top_ax.text(
            mid_point,
            top_ax.get_ylim()[1],  # Position at the top
            pb['PHASE'],
            ha='center',
            va='bottom',  # Align bottom of text to top of plot
            fontweight='bold',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='#222', alpha=0.7, ec='none'),
        )

    # Add action markers only to the top plot (axes[0, 0])
    action_steps = history_df[history_df['ACTION'].apply(lambda x: bool(x))]  # Filter steps with non-empty action lists
    if not action_steps.empty:
        # Calculate a Y position slightly above the bottom of the plot for visibility
        y_min, _y_max = top_ax.get_ylim()
        marker_y_pos = y_min

        # Plot a marker for each step with an action
        top_ax.scatter(
            action_steps['STEP'].unique(),
            [marker_y_pos] * len(action_steps['STEP'].unique()),
            marker='^',
            color='#aaa',
            s=40,
            zorder=10,
            label='Action Triggered',
        )

    # --- Update Legends ---
    # Add legend to each subplot
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        # Add the 'Action Triggered' marker to the top plot's legend if it exists
        if ax == top_ax and 'Action Triggered' in [h.get_label() for h in handles]:
            # Ensure 'Action Triggered' is handled correctly if present
            pass  # Already handled by get_legend_handles_labels
        elif 'Action Triggered' in labels:
            # Remove 'Action Triggered' from other plots if somehow added
            idx = labels.index('Action Triggered')
            handles.pop(idx)
            labels.pop(idx)

        if handles:  # Only add legend if there are items to show
            by_label = dict(zip(labels, handles, strict=True))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Set labels and title
    axes[0, 0].set_title('Timeline Property Evolution', fontsize=14)
    for ax in axes.flatten():
        ax.set_ylabel('Property value', fontsize=12)
    axes[-1, 0].set_xlabel('Step', fontsize=12)  # Set x-label only on the bottom plot

    # Add grid to allow key values to be compared
    for ax in axes.flatten():
        ax.grid(True, which='major', axis='both', linestyle=':', alpha=0.1)
        ax.margins(y=0.15)
    plt.tight_layout()
    return fig, axes
