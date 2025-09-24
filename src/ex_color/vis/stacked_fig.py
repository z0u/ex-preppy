from dataclasses import dataclass
from typing import Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from ex_color.data.color_cube import ColorCube
from ex_color.vis.plot_color_loss import draw_cube_series_on_ax
from ex_color.vis.plot_cube import draw_color_slice
from ex_color.vis.plot_latent_slices import ConicalAnnotation, LatentD, draw_latent_panel_from_cube
from utils.plt import Theme


@dataclass
class StackedFigure:
    fig: Figure
    ax_lat1: Axes3D
    ax_lat2: Axes3D
    ax_colors: Axes
    ax_loss: Axes


def build_stacked_figure(
    *,
    figsize=(10, 10),
    height_ratios: tuple[float, float, float] = (2.5, 2.0, 1.5),
) -> StackedFigure:
    """
    Create a 3-row stacked figure layout without drawing.

    Layout:
    - Top row: two 3D subplots (latents), side by side
    - Middle row: one wide 2D subplot (color slice)
    - Bottom row: one wide 2D subplot (loss series)

    Returns a StackedFigure containing the figure and axes.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=height_ratios)

    # Top: two 3D axes
    ax_lat1 = cast(Axes3D, fig.add_subplot(gs[0, 0], axes_class=Axes3D))
    ax_lat2 = cast(Axes3D, fig.add_subplot(gs[0, 1], axes_class=Axes3D))

    # Middle: full-width colors
    ax_colors = fig.add_subplot(gs[1, :])

    # Bottom: full-width loss
    ax_loss = fig.add_subplot(gs[2, :], sharex=ax_colors)

    return StackedFigure(fig=fig, ax_lat1=ax_lat1, ax_lat2=ax_lat2, ax_colors=ax_colors, ax_loss=ax_loss)


def draw_stacked_results(
    latent_cube: ColorCube,
    color_slice_cube: ColorCube,
    loss_cube: ColorCube,
    *,
    latent_dims: tuple[LatentD, LatentD],
    theme: Theme,
    max_error: float | None = None,
    latent_annotations: Sequence[ConicalAnnotation] = (),
):
    stack = build_stacked_figure(figsize=(5, 5.3), height_ratios=(2.5, 1.8, 1.2))
    # Top: latent space. Pick any two latent axis triplets you'd like to show
    for ax, dims in zip((stack.ax_lat1, stack.ax_lat2), latent_dims, strict=True):
        draw_latent_panel_from_cube(
            ax,
            latent_cube,
            dims=dims,
            colors='recon',
            colors_compare='color',
            latents='latents',
            dot_radius=5,
            theme=theme,
            annotations=latent_annotations,
        )
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-0.65, 0.65])
        ax.set_ylim([-0.65, 0.65])

    # Middle: reconstructed colors; pick a single slice index
    draw_color_slice(
        stack.ax_colors,
        color_slice_cube.permute('svh')[:, 1:, :],
        -1,  # Full saturation
        pretty=True,
        colors='recon',
        colors_compare='color',
    )
    stack.ax_colors.set_title('')
    stack.ax_colors.set_xlabel('')
    stack.ax_colors.xaxis.set_visible(False)

    # Bottom: loss vs. color series for a single cube variant
    draw_cube_series_on_ax(
        stack.ax_loss,
        loss_cube.permute('hsv')[:, -1:, :: (loss_cube.shape[2] // -5)],
        var='MSE',
    )
    stack.ax_loss.set_title('')
    stack.ax_loss.set_ylabel('MSE')
    stack.ax_loss.set_ylim(0, max_error)
    # format as :.2g
    # stack.ax_loss.yaxis.set_major_formatter('{x:.1g}')
    return stack.fig
