from itertools import product
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patheffects import Stroke

from ex_color.data.color import hues3, hues6, hues12
from ex_color.data.color_cube import ColorCube
from ex_color.vis.prettify import axname, prettify

# Accept either a single ColorCube or a sequence of ColorCubes


def _resolve_pretty(pretty: bool | str, space: str) -> str:
    if pretty is True:
        return space
    if pretty is False:
        return ''
    return str(pretty)


def _add_series_segments(
    ax: Axes, x_coords: np.ndarray, y: np.ndarray, seg_colors: np.ndarray, linewidth: float
) -> None:
    """Add colored line segments for a single Y series across X into the axes."""

    # Skip entirely NaN series
    if np.isnan(y).all():
        return

    xy0 = np.column_stack((x_coords[:-1], y[:-1]))
    xy1 = np.column_stack((x_coords[1:], y[1:]))
    if len(xy0) == 0:
        return
    # Drop segments containing NaNs
    mask = ~(np.isnan(xy0).any(axis=1) | np.isnan(xy1).any(axis=1))
    if not np.any(mask):
        return
    segs = np.stack((xy0[mask], xy1[mask]), axis=1)
    seg_colors = np.clip(seg_colors[mask], 0.0, 1.0)

    lc = LineCollection(
        list(segs),
        colors=seg_colors,
        linewidths=linewidth,
        alpha=1.0,
        path_effects=[Stroke(capstyle='round')],  # Round caps prevent gaps
    )
    ax.add_collection(lc)


def _format_axes_for_cube(ax: Axes, cube: ColorCube, var: str, *, ylim: tuple[float, float] | None = None, pretty: str):
    # X: show min/max (and maybe middle) to avoid clutter
    x_coords = cube.coordinates[0]
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_xlabel(axname(cube.space[0]).capitalize())

    if cube.space[0] == 'h':
        # Special case for hue
        domain = np.nanmax(x_coords) - np.nanmin(x_coords)
        hues = hues12 if domain <= 1 / 2 else hues6 if domain <= 2 / 3 else hues3
        tick_positions = list(hues.values())
        tick_labels = [h.capitalize() for h in hues.keys()]
        ax.set_xticks(tick_positions, tick_labels)
    elif x_coords[0] == 0 and x_coords[-1] == 1:
        ax.set_xticks([0, 1])
    else:
        # Choose sparse ticks: first, middle, last
        if len(x_coords) >= 3:
            mid_idx = len(x_coords) // 2
            ticks = [0, mid_idx, len(x_coords) - 1]
        else:
            ticks = list(range(len(x_coords)))
        tick_positions = [x_coords[i] for i in ticks]

        def fmt_val(axis: str, v: float | int) -> str:
            return prettify(float(v)) if axis in pretty else f'{float(v):.3g}'

        tick_labels = [fmt_val(cube.space[0], x_coords[i]) for i in ticks]
        ax.set_xticks(tick_positions, tick_labels)

    # Y limits with small margin
    y_min, y_max = ylim if ylim else (np.nanmin(cube[var]), np.nanmax(cube[var]))
    margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min, y_max + margin)
    if cast(Figure, ax.get_figure()).get_axes().index(ax) == 0:
        ax.set_ylabel(var.capitalize() if not var[0].isupper() else var)


def _plot_cube_on_axes(ax: Axes, cube: ColorCube, *, var: str, linewidth: float) -> None:
    """Render one cube's variable as colored segments into the provided axes."""
    data = cube[var]
    colors = cube['color']
    x, a, b = cube.coordinates

    for ia, ib in product(range(len(a)), range(len(b))):
        y = np.asarray(data[:, ia, ib])
        seg_colors = colors[:-1, ia, ib, :]
        _add_series_segments(ax, x, y, seg_colors, linewidth)


def draw_cube_series_on_ax(
    ax: Axes,
    cube: ColorCube,
    *,
    title: str | None = None,
    var: str,
    pretty: bool | str = True,
    linewidth: float = 1.4,
    y_min: float | None = None,
) -> Axes:
    """
    Draw a single cube series into an existing Axes.

    Draws colored line segments for the given scalar variable.
    Returns the axes for chaining.
    """
    _plot_cube_on_axes(ax, cube, var=var, linewidth=linewidth)
    y_max = float(np.nanmax(cube[var]))
    y_min_v = float(y_min) if y_min is not None else float(np.nanmin(cube[var]))
    _format_axes_for_cube(ax, cube, var, ylim=(y_min_v, y_max), pretty=_resolve_pretty(pretty, cube.space))
    if title:
        ax.set_title(title)
    return ax


def plot_cube_series(
    *cubes: ColorCube,
    title: str,
    var: str,
    pretty: bool | str = True,
    linewidth: float = 1.4,
    figsize: tuple[int, int] | None = None,
    y_min: float | None = None,
) -> Figure:
    """
    Plot scalar variable per color as colored line segments.

    Args:
        *cubes: One or more ColorCubes to plot side-by-side.
        title: Chart title.
        var: Name of the scalar variable to plot (e.g., 'loss').
        pretty: If True, prettify tick labels for all axes; if False, use raw numeric formatting. If a string, it specifies which axes to prettify, e.g., 'h' or 'hs'.
        linewidth: Width of the line segments.
        figsize: Figure size in inches.
        y_min: Optional minimum y-axis limit. If None, it is computed from the data.
    """
    if not cubes:
        raise ValueError('At least one ColorCube is required')

    y_min = cast(float, y_min if y_min is not None else min(np.nanmin(c[var]) for c in cubes))
    y_max = cast(float, max(np.nanmax(c[var]) for c in cubes))
    nx = [c.shape[0] for c in cubes]

    pretty = cubes[0].space if pretty is True else '' if pretty is False else pretty
    fig, axs = plt.subplots(1, len(cubes), figsize=figsize, width_ratios=nx, squeeze=False, sharey=True)
    _axs: list[Axes] = axs[0, :].tolist()
    for ax, cube in zip(_axs, cubes, strict=True):
        _plot_cube_on_axes(ax, cube, var=var, linewidth=linewidth)
        _format_axes_for_cube(ax, cube, var, ylim=(y_min, y_max), pretty=pretty)

    fig.suptitle(title, fontsize=10)

    return fig
