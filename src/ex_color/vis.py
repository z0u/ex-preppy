from dataclasses import dataclass
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import Stroke
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray

from ex_color.data.color_cube import ColorCube, color_axes
from utils.plt import Theme


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

    _title = f'{title} · ' if title else ''
    fig.suptitle(f'{_title}{y_axis.upper()} vs {x_axis.upper()} by {main_axis.upper()}')

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
            path_effects=[Stroke(capstyle='round')],  # Round caps prevent gaps between segments
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
    _title = f'{title} · ' if title else ''
    ax.set_title(
        f'{_title}{cube.canonical_space.upper()} loss vs {x_label}',
        fontsize=10,
    )

    plt.close()
    return fig


def _axname(axis_char: str) -> str:
    name = color_axes.get(axis_char.lower(), axis_char)
    return name


# Plotting utilities for latent slices


def draw_latent_3d(
    ax: Axes3D,
    latents_3d: np.ndarray,
    *,
    facecolors: np.ndarray,
    edgecolors: np.ndarray | None = None,
    dot_radius: float | np.ndarray = 5.0,
    linewidth_fraction: float = 0.4,
    alpha: float = 1.0,
):
    """
    Draw 3D scatter of latent points.

    Args:
        ax: Axes3D to draw on
        latents_3d: (N, 3)
        facecolors: (N, 3/4)
        edgecolors: optional (N, 3/4) for edge colors
        dot_radius: scalar or array of size (N,) to allow per-point sizing
        linewidth_fraction: fraction of dot_radius to use as linewidth
        alpha: overall alpha transparency for the points
    """
    if edgecolors is None:
        edgecolors = facecolors

    # Calculate linewidth as a fraction of the radius
    linewidths = linewidth_fraction * np.array(dot_radius)
    # The marker area (s) is in points^2, so reduce radius to keep stroke inside
    fill_radius = np.array(dot_radius) - linewidths
    fill_radius = np.clip(fill_radius, 0, None)
    s = np.pi * fill_radius**2

    scatter = ax.scatter(
        latents_3d[:, 0],
        latents_3d[:, 1],
        latents_3d[:, 2],  # type: ignore
        c=facecolors,
        edgecolors=cast(Sequence[ColorType], edgecolors),
        linewidths=linewidths,
        s=s,  # type: ignore
        alpha=alpha,
    )

    return {'scatter': scatter}


def draw_circle_3d(
    ax: Axes3D,
    r=1,
    *,
    verts=100,
    facecolor: ColorType | None = 'none',
    edgecolor: ColorType | None = 'none',
    linewidth: float | None = 1.0,
    zorder: float | None = None,
):
    theta = np.linspace(0, 2 * np.pi, verts)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, zorder or 0.0)

    # Create vertices for the filled circle
    circle_verts = [list(zip(x, y, z, strict=True))]
    facecolors = [facecolor] * len(circle_verts)
    edgecolors = [edgecolor] * len(circle_verts)
    poly = Poly3DCollection(circle_verts, facecolors=facecolors, edgecolors=edgecolors, linewidth=linewidth)
    ax.add_collection3d(poly)
    return poly


def draw_cone_3d(  # noqa: C901
    ax: Axes3D,
    *,
    direction: np.ndarray,  # [3]
    angle: float,
    **kwargs,
) -> list[Artist]:
    """
    Draw a minimal wireframe cone.

    The cone has tip at the origin and side length 1 (intersects the unit
    sphere). We draw only the natural edges in the current view:
    - The projected base ellipse (hidden if nearly edge-on).
    - Two silhouette generators from the tip, tangent to the ellipse.

    Notes
    -----
    - Camera is orthographic and axis-aligned: we project along +Z/-Z onto XY.
    - ``angle`` is the full spread (edge-to-edge); half-angle used internally.
    - ``direction`` must be shape (3,).
    """
    # Guard: invalid angle or zero directions
    if angle <= 0 or direction.size == 0:
        return []

    half = float(angle) * 0.5
    r = float(np.sin(half))
    c_scale = float(np.cos(half))

    # Visibility thresholds
    minor_axis_eps = 1e-3  # hide ellipse if its semi-minor < eps

    artists: list[Artist] = []

    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n

    def _basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Build an orthonormal basis (v, w) perpendicular to u
        a = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(u, a)) > 0.95:
            a = np.array([0.0, 1.0, 0.0])
        v = _unit(np.cross(u, a))
        w = _unit(np.cross(u, v))
        return v, w

    def _ellipse_axes_lengths(v2: np.ndarray, w2: np.ndarray) -> tuple[float, float]:
        # Semi-axes from SVD of mapping M = r * [v2 w2]
        M = r * np.column_stack((v2, w2))  # 2x2
        s = np.linalg.svd(M, compute_uv=False)
        # sorted descending
        return float(s[0]), float(s[1])

    def _proj_xy(p: np.ndarray) -> np.ndarray:
        return p[..., :2]

    def _p2(c2: np.ndarray, v2: np.ndarray, w2: np.ndarray, th: np.ndarray) -> np.ndarray:
        ct = np.cos(th)
        st = np.sin(th)
        return c2[None, :] + r * (ct[:, None] * v2[None, :] + st[:, None] * w2[None, :])

    def _p2_d1(v2: np.ndarray, w2: np.ndarray, th: np.ndarray) -> np.ndarray:
        ct = np.cos(th)
        st = np.sin(th)
        return r * (-st[:, None] * v2[None, :] + ct[:, None] * w2[None, :])

    def _p2_d2(v2: np.ndarray, w2: np.ndarray, th: np.ndarray) -> np.ndarray:
        ct = np.cos(th)
        st = np.sin(th)
        return r * (-ct[:, None] * v2[None, :] - st[:, None] * w2[None, :])

    def _find_tangent_thetas(c2: np.ndarray, v2: np.ndarray, w2: np.ndarray) -> list[float]:  # noqa: C901
        # Solve f(θ) = cross2d(p(θ), p'(θ)) = 0 for tangency
        # (origin lies on the tangent line when p and p' are colinear)
        # Robust search + bracketed refinement
        def f(th: np.ndarray) -> np.ndarray:
            p = _p2(c2, v2, w2, th)
            dp = _p2_d1(v2, w2, th)
            return p[:, 0] * dp[:, 1] - p[:, 1] * dp[:, 0]

        def fp(th: np.ndarray) -> np.ndarray:
            p = _p2(c2, v2, w2, th)
            ddp = _p2_d2(v2, w2, th)
            # derivative of cross(p, p') is cross(p, p'')
            return p[:, 0] * ddp[:, 1] - p[:, 1] * ddp[:, 0]

        # Coarse sampling
        N = 720
        ths = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        vals = f(ths)
        roots: list[float] = []

        # Bracket sign changes
        for i in range(N):
            i2 = (i + 1) % N
            a, b = ths[i], ths[i2]
            fa, fb = vals[i], vals[i2]
            if fa == 0.0:
                roots.append(a)
            elif fa * fb < 0.0:
                # Bisection + Newton refinement
                lo, hi = a, b
                flo = fa
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    fm = f(np.array([mid]))[0]
                    if flo * fm <= 0:
                        hi = mid
                    else:
                        lo, flo = mid, fm
                th0 = 0.5 * (lo + hi)
                # Newton (guarded)
                for _ in range(6):
                    fv = f(np.array([th0]))[0]
                    df = fp(np.array([th0]))[0]
                    if df == 0:
                        break
                    step = fv / df
                    th1 = th0 - step
                    # keep within [0, 2π)
                    th1 = (th1 + 2.0 * np.pi) % (2.0 * np.pi)
                    th0 = th1
                roots.append(th0)

        if len(roots) == 0:
            # Fallback: take two minima of |f| separated sufficiently
            idx = np.argsort(np.abs(vals))[:8]
            cand = sorted(float(ths[i]) for i in idx)
            pruned: list[float] = []
            for th in cand:
                if all(abs(((th - x + np.pi) % (2 * np.pi)) - np.pi) > 0.1 for x in pruned):
                    pruned.append(th)
                if len(pruned) == 2:
                    break
            roots = pruned

        # Deduplicate and keep two opposite-ish solutions
        uniq: list[float] = []
        for th in sorted(roots):
            if all(abs(((th - x + np.pi) % (2 * np.pi)) - np.pi) > 1e-2 for x in uniq):
                uniq.append(th)
        if len(uniq) > 2:
            # pick two that are most separated
            best = (uniq[0], uniq[1])
            best_sep = 0.0
            for a in uniq:
                for b in uniq:
                    d = abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)
                    if d > best_sep:
                        best, best_sep = (a, b), d
            uniq = [best[0], best[1]]
        # if len(uniq) < 2:
        #     uniq = [uniq[0], uniq[0]]
        return uniq

    def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
        # Ray casting algorithm; poly is (M, 2)
        x, y = pt
        inside = False
        n = poly.shape[0]
        xj, yj = poly[-1]
        for i in range(n):
            xi, yi = poly[i]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
                if x < x_int:
                    inside = not inside
            xj, yj = xi, yi
        return inside

    u = _unit(direction)
    if np.linalg.norm(u) == 0.0:
        # degenerate
        raise ValueError('direction cannot be the zero vector')

    v, w = _basis(u)
    c = c_scale * u  # circle center on sphere
    c2 = _proj_xy(c)
    v2 = _proj_xy(v)
    w2 = _proj_xy(w)

    # Ellipse axes for visibility decision
    a_len, b_len = _ellipse_axes_lengths(v2, w2)
    show_ellipse = b_len >= minor_axis_eps

    # Build dense ellipse samples (projection of base circle)
    theta = np.linspace(0.0, 2.0 * np.pi, 361)

    def _draw_ellipse():
        if show_ellipse:
            circle_pts = c[None, :] + r * (np.cos(theta)[:, None] * v[None, :] + np.sin(theta)[:, None] * w[None, :])
            # Draw dashed/dotted ellipse (projected circle)
            (el,) = ax.plot(
                circle_pts[:, 0],
                circle_pts[:, 1],
                circle_pts[:, 2],  # type: ignore
                zorder=50,
                **kwargs,
            )
            artists.append(el)

    # Tangent edges (skip if tip inside the ellipse)
    def _draw_edges():
        # Determine if the tip projects inside the ellipse
        poly2 = _p2(c2, v2, w2, theta)
        if not _point_in_poly(np.array([0.0, 0.0]), poly2):
            a, b = _find_tangent_thetas(c2, v2, w2)
            pa3 = c + r * (np.cos(a) * v + np.sin(a) * w)
            pb3 = c + r * (np.cos(b) * v + np.sin(b) * w)
            xs, ys, zs = zip(pa3, (0.0, 0.0, 0.0), pb3, strict=True)
            (ln,) = ax.plot(
                xs,
                ys,
                zs,  # type: ignore
                zorder=50,
                **kwargs,
            )
            artists.append(ln)

    if direction[2] < 0:
        _draw_edges()
        _draw_ellipse()
    else:
        _draw_ellipse()
        _draw_edges()

    return artists


@dataclass
class ConicalAnnotation:
    direction: np.ndarray | Sequence[float]  # [3]
    angle: float
    line_kwargs: dict[str, Any]

    def __init__(
        self,
        direction: np.ndarray | Sequence[float],  # [3]
        angle: float,
        **line_kwargs,
    ):
        self.direction = direction
        self.angle = angle
        self.line_kwargs = line_kwargs


def plot_latent_grid_3d(
    latents: torch.Tensor,
    colors: torch.Tensor,
    colors_compare: torch.Tensor | None = None,
    *,
    dims: list[tuple[int, int, int]],
    title: str | None = None,
    figsize_per_plot: tuple[float, float] = (6, 6),
    dot_radius: float = 10.0,
    theme: Theme,
    annotations: Sequence[ConicalAnnotation] | None = None,
):
    """Plot 4D+ latent data as 3D visualizations."""
    lat_np = latents.detach().cpu().numpy()
    col_np = colors.detach().cpu().reshape(-1, colors.shape[-1]).numpy()
    col_compare_np = (
        colors_compare.detach().cpu().reshape(-1, colors.shape[-1]).numpy() if colors_compare is not None else col_np
    )

    n = len(dims)
    cols = min(3, n)  # Max 3 columns
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows), constrained_layout=True)

    for idx, (i, j, k) in enumerate(dims):
        # Create 3D subplot
        ax = cast(Axes3D, fig.add_subplot(rows, cols, idx + 1, axes_class=Axes3D))

        # Extract 3D coordinates
        lat_3d = lat_np[:, [i, j, k]]

        draw_circle_3d(ax, facecolor=theme.val('#8888', dark='#111', light='#eee'), zorder=-10)
        draw_latent_3d(ax, lat_3d, edgecolors=col_np, facecolors=col_compare_np, alpha=1.0, dot_radius=dot_radius)
        draw_circle_3d(ax, edgecolor='#0005', linewidth=1, zorder=10)
        for a in annotations or []:
            direction = np.asarray(a.direction)[[i, j, k]]
            draw_cone_3d(ax, direction=direction, angle=a.angle, **a.line_kwargs)

        # Clean up the 3D axes
        ax.set_axis_off()
        ax.patch.set_alpha(0)
        ax.set_title(f'({i},{j},{k})')
        # Always look downwards from the "top": the axis ordering (i,j,k) determines the view
        ax.view_init(elev=90, azim=-90)
        ax.set_proj_type('ortho')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    if title:
        fig.suptitle(title)

    return fig
