from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ex_color.data.color import hues3, hues6, hues12
from ex_color.data.color_cube import ColorCube
from ex_color.vis.prettify import axname, prettify


def _resolve_pretty(pretty: bool | str, space: str) -> str:
    if pretty is True:
        return space
    if pretty is False:
        return ''
    return str(pretty)


def _coerce_colors_arg(cube: ColorCube, colors: np.ndarray | str | None) -> np.ndarray:
    if colors is None:
        return cube.rgb_grid
    if isinstance(colors, str):
        return cube[colors]
    return colors


def annotate_cells(ax: Axes, b: np.ndarray):
    """Draw a colored rectangle per cell using colors_compare."""
    from matplotlib.patches import Rectangle

    H, W = b.shape[:2]
    # Ensure axis limits correspond to the pixel grid
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    width = 0.3
    half_width = width / 2
    for r in range(H):
        for c in range(W):
            rect = Rectangle(
                (c - half_width, r - half_width),
                width,
                width,
                facecolor=b[r, c],
            )
            ax.add_patch(rect)


def draw_color_slice(
    ax: Axes,
    cube: ColorCube,
    i: int,
    *,
    pretty: bool | str = True,
    colors: np.ndarray | str | None = None,
    colors_compare: np.ndarray | str | None = None,
):
    """Draw one 2D slice (at main index i) of a ColorCube into ax."""
    pretty = _resolve_pretty(pretty, cube.space)

    def fmt(axis: str, v: float | int) -> str:
        if axis in pretty:
            return prettify(float(v))
        else:
            return f'{v:.2g}'

    main_axis, y_axis, x_axis = cube.space
    main_coords, y_coords, x_coords = cube.coordinates

    colors = _coerce_colors_arg(cube, colors)
    colors_compare = None if colors_compare is None else _coerce_colors_arg(cube, colors_compare)

    ax.imshow(colors[i], vmin=0, vmax=1)
    if colors_compare is not None:
        annotate_cells(ax, colors_compare[i])

    ax.set_aspect('equal')
    ax.set_title(f'{axname(main_axis).capitalize()} = {fmt(main_axis, main_coords[i])}')

    # Ticks/labels
    if x_axis in pretty:
        ax.xaxis.set_ticks(*_get_pretty_ticks(x_axis, x_coords))
    else:
        ax.xaxis.set_ticks(*_get_plain_ticks(x_coords))
    ax.set_xlabel(axname(x_axis).capitalize())

    if y_axis in pretty:
        ax.yaxis.set_ticks(*_get_pretty_ticks(y_axis, y_coords))
    else:
        ax.yaxis.set_ticks(*_get_plain_ticks(y_coords))
    ax.set_ylabel(axname(y_axis).capitalize())

    return ax


def plot_colors(  # noqa: C901
    cube: ColorCube,
    pretty: bool | str = True,
    patch_size: float = 0.25,
    title: str | None = '',
    colors: np.ndarray | str | None = None,
    colors_compare: np.ndarray | str | None = None,
):
    """Plot a ColorCube in 2D slices."""
    from itertools import chain
    from math import ceil

    main_axis, y_axis, x_axis = cube.space
    main_coords, y_coords, x_coords = cube.coordinates

    n_plots = len(main_coords)
    nominal_width = 70
    full_width = len(x_coords) * n_plots + (n_plots - 1)
    n_rows = ceil(full_width / nominal_width)
    n_cols = ceil(n_plots / n_rows)

    # Base size per subplot, adjusted by the data dimensions
    subplot_width = patch_size * len(x_coords)
    subplot_height = patch_size * len(y_coords) + 0.5
    figsize = (n_cols * subplot_width, n_rows * subplot_height)

    axes: Sequence[Axes] | NDArray
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = list(chain(*axes))

    for idx, ax in enumerate(axes):
        if idx >= len(main_coords):
            ax.set_visible(False)
            continue

        draw_color_slice(ax, cube, idx, pretty=pretty, colors=colors, colors_compare=colors_compare)

    if title is not None:
        _title = f'{title} · ' if title else ''
        fig.suptitle(f'{_title}{y_axis.upper()} vs {x_axis.upper()} by {main_axis.upper()}')

    plt.close()
    return fig


def _get_plain_ticks(values: np.ndarray):
    return [0, len(values) - 1], [values[0], values[-1]]


def _get_pretty_ticks(axis: str, values: np.ndarray):
    if axis == 'h':
        idxs, labels = _get_close_hue_ticks(values)
        if idxs:
            return idxs, labels
    return [0, len(values) - 1], [prettify(values[0]), prettify(values[-1])]


def _get_close_hue_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    close_x_idxs = None
    names = None
    cluse_hue_idxs = None

    for hues in [hues3, hues6, hues12]:
        names = [h.capitalize() for h in hues.keys()]
        close_x_idxs, close_hue_idxs = np.where(np.isclose(values[:, None], list(hues.values())))
        if len(close_x_idxs) >= 2:
            return (close_x_idxs.tolist(), np.array(names)[close_hue_idxs].tolist())

    assert close_x_idxs is not None
    assert names is not None
    assert cluse_hue_idxs is not None
    return (close_x_idxs.tolist(), np.array(names)[cluse_hue_idxs].tolist())
