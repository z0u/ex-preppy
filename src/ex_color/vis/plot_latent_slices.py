from dataclasses import dataclass
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.artist import Artist
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ex_color.data.color_cube import ColorCube
from utils.plt import Theme


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


def plot_latent_grid_3d_from_cube(
    cube: ColorCube,
    colors: str = 'color',
    colors_compare: str | None = None,
    latents: str = 'latents',
    *,
    dims: Sequence[tuple[int, int, int]],
    title: str | None = None,
    figsize_per_plot: tuple[float, float] = (6, 6),
    dot_radius: float = 10.0,
    theme: Theme,
    annotations: Sequence[ConicalAnnotation] | None = None,
):
    return plot_latent_grid_3d(
        cube[latents],
        cube[colors],
        cube[colors_compare] if colors_compare is not None else None,
        dims=dims,
        title=title,
        figsize_per_plot=figsize_per_plot,
        dot_radius=dot_radius,
        theme=theme,
        annotations=annotations,
    )


def draw_latent_panel(
    ax: Axes3D,
    latents: torch.Tensor | np.ndarray,
    colors: torch.Tensor | np.ndarray,
    colors_compare: torch.Tensor | np.ndarray | None,
    *,
    dims: tuple[int, int, int],
    dot_radius: float,
    theme: Theme,
    annotations: Sequence[ConicalAnnotation] | None = None,
    title: str | None = None,
):
    """
    Draw a single 3D latent slice into the provided Axes3D.

    Contract
    - Inputs: latents [..., D], colors [..., 3], colors_compare [..., 3] or None; dims (i,j,k)
    - Output: draws on ax; returns ax
    - Errors: raises if dims out of range
    """
    if colors_compare is None:
        colors_compare = colors

    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    if isinstance(colors_compare, torch.Tensor):
        colors_compare = colors_compare.detach().cpu().numpy()

    # flatten all but last dim
    latents = latents.reshape(-1, latents.shape[-1])
    colors = colors.reshape(-1, colors.shape[-1])
    colors_compare = colors_compare.reshape(-1, colors_compare.shape[-1])

    i, j, k = dims
    lat_3d = latents[:, [i, j, k]]

    draw_circle_3d(ax, facecolor=theme.val('#8888', dark='#111', light='#eee'), zorder=-10)
    draw_latent_3d(
        ax,
        lat_3d,
        edgecolors=colors,
        facecolors=colors_compare,
        alpha=1.0,
        dot_radius=dot_radius,
    )
    draw_circle_3d(ax, edgecolor='#0005', linewidth=1, zorder=10)
    for a in annotations or []:
        direction = np.asarray(a.direction)[[i, j, k]]
        draw_cone_3d(ax, direction=direction, angle=a.angle, **a.line_kwargs)

    # Clean up the 3D axes
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    if title:
        ax.set_title(title)
    # Always look downwards from the "top": the axis ordering (i,j,k) determines the view
    ax.view_init(elev=90, azim=-90)
    ax.set_proj_type('ortho')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    return ax


def draw_latent_panel_from_cube(
    ax: Axes3D,
    cube: ColorCube,
    *,
    dims: tuple[int, int, int],
    colors: str = 'color',
    colors_compare: str | None = None,
    latents: str = 'latents',
    dot_radius: float = 10.0,
    theme: Theme,
    annotations: Sequence[ConicalAnnotation] | None = None,
    title: str | None = None,
):
    """Draw a single 3D latent panel from a ColorCube into ax."""
    return draw_latent_panel(
        ax,
        cube[latents],
        cube[colors],
        cube[colors_compare] if colors_compare is not None else None,
        dims=dims,
        dot_radius=dot_radius,
        theme=theme,
        annotations=annotations,
        title=title,
    )


def plot_latent_grid_3d(
    latents: torch.Tensor | np.ndarray,
    colors: torch.Tensor | np.ndarray,
    colors_compare: torch.Tensor | np.ndarray | None = None,
    *,
    dims: Sequence[tuple[int, int, int]],
    title: str | None = None,
    figsize_per_plot: tuple[float, float] = (6, 6),
    dot_radius: float = 10.0,
    theme: Theme,
    annotations: Sequence[ConicalAnnotation] | None = None,
):
    """Plot 4D+ latent data as 3D visualizations."""
    if colors_compare is None:
        colors_compare = colors

    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()  # .reshape(-1, colors.shape[-1]).numpy()
    if isinstance(colors_compare, torch.Tensor):
        colors_compare = colors_compare.detach().cpu().numpy()  # .reshape(-1, colors.shape[-1]).numpy()

    # flatten all but last dim
    latents = latents.reshape(-1, latents.shape[-1])
    colors = colors.reshape(-1, colors.shape[-1])
    colors_compare = colors_compare.reshape(-1, colors_compare.shape[-1])

    n = len(dims)
    cols = min(3, n)  # Max 3 columns
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows), constrained_layout=True)

    for idx, dims_ijk in enumerate(dims):
        # Create 3D subplot and delegate drawing
        ax = cast(Axes3D, fig.add_subplot(rows, cols, idx + 1, axes_class=Axes3D))
        draw_latent_panel(
            ax,
            latents,
            colors,
            colors_compare,
            dims=dims_ijk,
            dot_radius=dot_radius,
            theme=theme,
            annotations=annotations,
            title=f'({dims_ijk[0]},{dims_ijk[1]},{dims_ijk[2]})',
        )

    if title:
        fig.suptitle(title)

    return fig
