from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline
from mini.temporal.vis import plot_timeline, realize_timeline, ParamGroup
from utils.plt import Theme


def plot_dopesheet(dopesheet: Dopesheet, theme: Theme):
    timeline = Timeline(dopesheet)
    history_df = realize_timeline(timeline)
    keyframes_df = dopesheet.as_df()

    fig = plt.figure(figsize=(9, 3), constrained_layout=True)
    axs = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    ax1, ax2 = cast(tuple[Axes, ...], axs)

    plot_timeline(
        history_df,
        keyframes_df,
        groups=(ParamGroup(name='', params=[p for p in dopesheet.props if p not in {'lr'}]),),
        theme=theme,
        ax=ax1,
        show_phase_labels=False,
    )
    ax1.set_ylabel('Weight')
    ax1.set_xlabel('')

    plot_timeline(
        history_df,
        keyframes_df,
        groups=(ParamGroup(name='', params=['lr']),),
        theme=theme,
        ax=ax2,
        show_legend=False,
        show_phase_labels=False,
    )
    ax2.set_ylabel('LR')

    # add a little space on the y-axis extents
    ax1.set_ylim(ax1.get_ylim()[0] * 1.1, ax1.get_ylim()[1] * 1.1)
    ax2.set_ylim(ax2.get_ylim()[0] * 1.1, ax2.get_ylim()[1] * 1.1)

    return fig
