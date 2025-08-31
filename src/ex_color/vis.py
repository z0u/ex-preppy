from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ex_color.data.color_cube import ColorCube, color_axes


def plot_colors(  # noqa: C901
    cube: ColorCube,
    pretty: bool | str = True,
    patch_size: float = 0.25,
    title: str = '',
    colors: np.ndarray | None = None,
    colors_compare: np.ndarray | None = None,
):
    """Plot a ColorCube in 2D slices."""
    from itertools import chain
    from math import ceil

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if pretty is True:
        pretty = cube.space
    elif pretty is False:
        pretty = ''

    def fmt(axis: str, v: float | int) -> str:
        if axis in pretty:
            return prettify(float(v))
        else:
            return f'{v:.2g}'

    # Create a figure with subplots

    main_axis, y_axis, x_axis = cube.space
    main_coords, y_coords, x_coords = cube.coordinates

    n_plots = len(main_coords)
    nominal_width = 70
    full_width = len(x_coords) * n_plots + (n_plots - 1)
    n_rows = ceil(full_width / nominal_width)
    n_cols = ceil(n_plots / n_rows)

    # Calculate appropriate figure size based on data dimensions
    # Base size per subplot, adjusted by the data dimensions
    subplot_width = patch_size * len(x_coords)
    subplot_height = patch_size * len(y_coords) + 0.5

    # Calculate total figure size with some margins between plots
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
    axes = list(chain(*axes))  # Flatten the axes array

    if colors is None:
        colors = cube.rgb_grid

    def annotate_cells(ax: Axes, b: np.ndarray):
        """
        Draw a colored outline rectangle per cell using colors_compare.

        edge_colors shape: (H, W, 3) in [0, 1].
        """
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

    # Plot each slice of the cube (one for each value)
    for i, ax in enumerate(axes):
        if i >= len(main_coords):
            ax.set_visible(False)
            continue
        row = i // n_cols
        col = i % n_cols

        ax.imshow(colors[i], vmin=0, vmax=1)
        if colors_compare is not None:
            annotate_cells(ax, colors_compare[i])

        ax.set_aspect('equal')
        ax.set_title(f'{_axname(main_axis).capitalize()} = {fmt(main_axis, main_coords[i])}', fontsize=8)

        # Add axes labels without cluttering the display
        if row == n_rows - 1:
            ax.xaxis.set_ticks([0, len(x_coords) - 1])
            coord1 = fmt(x_axis, x_coords[0])
            coord2 = fmt(x_axis, x_coords[-1])
            ax.xaxis.set_ticklabels([coord1, coord2])
            ax.xaxis.set_tick_params(labelsize=8)
            ax.set_xlabel(_axname(x_axis).capitalize(), fontsize=8)
        else:
            ax.xaxis.set_visible(False)

        if col == 0:
            ax.yaxis.set_ticks([0, len(y_coords) - 1])
            coord1 = fmt(y_axis, y_coords[0])
            coord2 = fmt(y_axis, y_coords[-1])
            ax.yaxis.set_ticklabels([coord1, coord2])
            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_ylabel(_axname(y_axis).capitalize(), fontsize=8)
        else:
            ax.yaxis.set_visible(False)

    _title = f'{title} - ' if title else ''
    plt.suptitle(
        f'{_title}{cube.canonical_space.upper()} as {y_axis.upper()} vs {x_axis.upper()} by {main_axis.upper()}'
    )

    plt.close()
    return fig


def prettify(value: float, tolerance=1e-10):
    """Convert a float to a string, attempting to make it more human-readable."""
    from sympy import nsimplify, pretty

    # result = fu(value)
    result = nsimplify(value, tolerance=tolerance, rational_conversion='exact')
    s = pretty(result, use_unicode=True)
    if '\n' in s:
        return str(result)
    else:
        return s


def plot_loss_lines(  # noqa: C901
    cube: ColorCube,
    title: str,
    loss: np.ndarray,
    *,
    ylabel: str = 'Loss',
    colors: np.ndarray | None = None,
    pretty: bool | str = True,
    linewidth: float = 1.4,
    figsize: tuple[int, int] | None = (12, 3),
) -> Figure:
    """
    Plot reconstruction loss per color as colored line segments.

    The x-axis is the first axis of the cube's canonical space (e.g., H for HSV).
    Each line corresponds to a pair of coordinates from the remaining two axes.
    Line color follows the true colors provided via ``colors`` (RGB in [0, 1]).

    Parameters
    ----------
    loss : np.ndarray
        Array shaped like the cube's grid in ``cube.space`` order, with one
        scalar loss per color (no channel). This function does not compute loss.
    cube : ColorCube
        Color cube whose coordinates define axes and ordering.
    title : str, default ''
        Optional chart title prefix.
    ylabel : str
        Y-axis label.
    colors : np.ndarray | None, default None
        True colors as RGB floats in [0, 1], shaped like the cube's grid with a
        trailing channel, i.e., (..., 3). Defaults to ``cube.rgb_grid``.
    pretty : bool | str, default True
        If True, prettify tick labels for all axes; if False, use raw numeric
        formatting. If a string, it specifies which axes to prettify, e.g.,
        'h' or 'hs'.
    linewidth : float, default 1.4
        Width of the line segments.
    figsize : tuple[int, int] | None, default (9, 4)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure containing the plot.
    """
    from itertools import product
    from math import isnan

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # Resolve pretty axes selection string
    if pretty is True:
        pretty = cube.space
    elif pretty is False:
        pretty = ''

    # Validate and default colors
    if colors is None:
        colors = cube.rgb_grid

    # Map current space ordering to canonical ordering
    # cube.space is the ordering of the grid dimensions (e.g., 'svh') used in arrays
    # cube.canonical_space is the semantic ordering (e.g., 'hsv')
    space = tuple(cube.space)
    canon = tuple(cube.canonical_space)

    axis_to_dim = {axis: i for i, axis in enumerate(space)}
    axis_to_coords = dict(zip(space, cube.coordinates, strict=True))

    # Build permutation to transpose arrays from space->canonical order
    perm = [axis_to_dim[canon[0]], axis_to_dim[canon[1]], axis_to_dim[canon[2]]]

    # Bring arrays into canonical order: (X, A, B) where X is canon[0]
    loss_c = np.transpose(loss, perm)
    if colors.ndim == 4:
        colors_c = np.transpose(colors, perm + [3])
    else:
        raise ValueError('colors must be an array of shape (..., 3) with RGB channels')

    x_coords = np.asarray(axis_to_coords[canon[0]])
    a_coords = np.asarray(axis_to_coords[canon[1]])
    b_coords = np.asarray(axis_to_coords[canon[2]])

    # Figure sizing based on resolution
    n_x = len(x_coords)
    n_a = len(a_coords)
    n_b = len(b_coords)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Build line segments per (a, b) series; color per segment using true color at start point
    y_min = np.nanmin(loss_c)
    y_max = np.nanmax(loss_c)
    if isnan(y_min) or isnan(y_max):  # guard
        y_min, y_max = 0.0, 1.0

    for ia, ib in product(range(n_a), range(n_b)):
        y = np.asarray(loss_c[:, ia, ib])
        # Skip entirely nan series
        if np.isnan(y).all():
            continue
        # Build segments: shape (n_segments, 2, 2)
        xy0 = np.column_stack((x_coords[:-1], y[:-1]))
        xy1 = np.column_stack((x_coords[1:], y[1:]))
        if len(xy0) == 0:
            continue
        # Drop segments with NaNs
        mask = ~(np.isnan(xy0).any(axis=1) | np.isnan(xy1).any(axis=1))
        if not np.any(mask):
            continue
        segs = np.stack((xy0[mask], xy1[mask]), axis=1)
        segs_list = list(segs)
        seg_colors = colors_c[:-1, ia, ib, :][mask]
        # Clamp to [0, 1]
        seg_colors = np.clip(seg_colors, 0.0, 1.0)
        lc = LineCollection(
            segs_list,
            colors=seg_colors,
            linewidths=linewidth,
            alpha=1.0,
        )
        ax.add_collection(lc)

    # Axes formatting
    # X: show min/max (and maybe middle) to avoid clutter
    ax.set_xlim(float(x_coords[0]), float(x_coords[-1]))
    x_label = _axname(canon[0])
    ax.set_xlabel(x_label.capitalize())

    # Choose sparse ticks: first, middle, last
    if n_x >= 3:
        mid_idx = n_x // 2
        ticks = [0, mid_idx, n_x - 1]
    else:
        ticks = list(range(n_x))
    tick_positions = [float(x_coords[i]) for i in ticks]

    def fmt_val(axis: str, v: float | int) -> str:
        return prettify(float(v)) if axis in pretty else f'{float(v):.3g}'

    tick_labels = [fmt_val(canon[0], x_coords[i]) for i in ticks]
    ax.set_xticks(tick_positions, tick_labels)

    # Y limits with small margin
    margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min, y_max + margin)
    ax.set_ylabel(ylabel)

    # Title
    _title = f'{title} - ' if title else ''
    ax.set_title(
        f'{_title}{cube.canonical_space.upper()} loss vs {x_label}',
        fontsize=10,
    )

    plt.close()
    return fig


def _axname(axis_char: str) -> str:
    name = color_axes.get(axis_char.lower(), axis_char)
    return name
