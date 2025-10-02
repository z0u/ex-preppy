from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from ex_color.data import ColorCube
from utils.plt import Theme


def draw_cube_scatter(
    ax: Axes,
    cube: ColorCube,
    *,
    theme: Theme,
    x_var: str,
    y_var: str,
    color_var: str | None = 'color',
    regression: bool = True,
):
    """Scatter plot of two variables from a color cube."""
    from scipy.stats import linregress

    x = cube[x_var].flatten()
    y = cube[y_var].flatten()
    color = cube[color_var].reshape((-1, 3)) if color_var else None

    ax.scatter(
        x,
        y,
        alpha=((x + y) / 2) * 0.9 + 0.1,  # type: ignore[reportArgumentType]
        s=2,
        color=color,
    )
    if regression:
        slope, intercept, r = linregress(x, y)[:3]
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            '--',
            color=theme.val('black', dark='white'),
            gapcolor=theme.val('#fff8', dark='#222a'),
            linewidth=1,
            label=f'$R^2 = {r**2:.3f}$',
        )


def plot_cube_scatter(
    cube: ColorCube,
    *,
    theme: Theme,
    x_var: str,
    y_var: str,
    color_var: str | None = 'color',
    regression: bool = True,
    figsize: tuple[float, float] = (4, 4),
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_cube_scatter(
        ax,
        cube,
        theme=theme,
        x_var=x_var,
        y_var=y_var,
        color_var=color_var,
        regression=regression,
    )
    ax.set_xlabel(x_var.replace('_', ' ').capitalize())
    ax.set_ylabel(y_var.replace('_', ' ').capitalize())
    ax.legend(loc='upper left')
    return fig
