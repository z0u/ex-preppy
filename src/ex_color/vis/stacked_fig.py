from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class StackedFigure:
    fig: Figure
    ax_lat1: Axes3D
    ax_lat2: Axes3D
    ax_colors: Axes
    ax_loss: Axes


def build_stacked_figure(*, figsize=(10, 10)) -> StackedFigure:
    """
    Create a 3-row stacked figure layout without drawing.

    Layout:
    - Top row: two 3D subplots (latents), side by side
    - Middle row: one wide 2D subplot (color slice)
    - Bottom row: one wide 2D subplot (loss series)

    Returns a StackedFigure containing the figure and axes.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.8])

    # Top: two 3D axes
    ax_lat1 = cast(Axes3D, fig.add_subplot(gs[0, 0], axes_class=Axes3D))
    ax_lat2 = cast(Axes3D, fig.add_subplot(gs[0, 1], axes_class=Axes3D))

    # Middle: full-width colors
    ax_colors = fig.add_subplot(gs[1, :])

    # Bottom: full-width loss
    ax_loss = fig.add_subplot(gs[2, :])

    return StackedFigure(fig=fig, ax_lat1=ax_lat1, ax_lat2=ax_lat2, ax_colors=ax_colors, ax_loss=ax_loss)
