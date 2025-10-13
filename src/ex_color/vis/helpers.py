"""
Helper functions for common visualization patterns in color experiments.

This module provides reusable visualization functions for creating standardized
plots across multiple experiment notebooks.
"""

import re
from collections.abc import Callable, Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mini.temporal import Dopesheet, ParamGroup, Timeline, plot_timeline, realize_timeline
from ex_color.data.color_cube import ColorCube
from ex_color.evaluation import Resultset
from ex_color.vis import ConicalAnnotation, draw_cube_scatter, draw_stacked_results, plot_colors, plot_cube_series
from ex_color.vis.plot_latent_slices import LatentD
from utils.nb import displayer_mpl
from utils.plt import Theme


def tags_for_file(tags: Sequence[str]) -> str:
    """
    Convert a sequence of tags into a filename-safe string.

    Args:
        tags: Sequence of tag strings

    Returns:
        Hyphen-separated string with only alphanumeric characters and hyphens
    """
    tags = [re.sub(r'[^a-zA-Z0-9]+', '-', tag.lower()) for tag in tags]
    return '-'.join(tags)


def visualize_reconstructed_cube(
    data: ColorCube,
    *,
    tags: Sequence[str] = (),
    nbid: str,
) -> None:
    """
    Visualize reconstructed colors from a model.

    Creates a figure showing slices of the HSV cube with reconstructed colors
    compared to true colors.

    Args:
        data: ColorCube with 'recon' and 'color' variables
        tags: Tags to include in filename and title
        nbid: Notebook ID for filename prefix
    """
    with displayer_mpl(
        f'large-assets/ex-{nbid}-pred-colors-{tags_for_file(tags)}.png',
        alt_text="""Plot showing four slices of the HSV cube, titled "{title}". Nominally, each slice has constant saturation, but varies in value (brightness) from top to bottom, and in hue from left to right. Each color value is represented as a square patch of that color. The outer portion of the patches shows the color as reconstructed by the model; the inner portion shows the true (input) color.""",
    ) as show:
        show(
            lambda: plot_colors(
                data,
                title=f'Predicted colors · {" · ".join(tags)}',
                colors='recon',
                colors_compare='color',
            )
        )


def visualize_reconstruction_loss(
    data: ColorCube,
    *,
    tags: Sequence[str] = (),
    nbid: str,
) -> None:
    """
    Visualize reconstruction loss across color space.

    Creates line charts showing MSE loss patterns across different
    slices of the color cube.

    Args:
        data: ColorCube with 'MSE' variable
        tags: Tags to include in filename and title
        nbid: Notebook ID for filename prefix
    """
    max_loss = np.max(data['MSE'])
    median_loss = np.median(data['MSE'])

    with displayer_mpl(
        f'large-assets/ex-{nbid}-loss-colors-{tags_for_file(tags)}.png',
        alt_text=f"""Line chart showing loss per color, titled "{{title}}". Y-axis: mean square error, ranging from zero to {max_loss:.2g}. X-axis: hue.""",
    ) as show:
        show(
            lambda: plot_cube_series(
                data.permute('hsv')[:, -1:, :: (data.shape[2] // -5)],
                data.permute('svh')[:, -1:, :: -(data.shape[0] // -3)],
                data.permute('vsh')[:, -1:, :: -(data.shape[0] // -3)],
                title=f'Reconstruction error · {" · ".join(tags)}',
                var='MSE',
                figsize=(12, 3),
            )
        )
    print(f'Max loss: {max_loss:.2g}')
    print(f'Median loss: {median_loss:.2g}')


def visualize_stacked_results(
    resultset: Resultset,
    *,
    latent_dims: tuple[LatentD, LatentD],
    max_error: float | None = None,
    latent_annotations: Sequence[ConicalAnnotation | Callable[[Theme], ConicalAnnotation]] = (),
    nbid: str,
) -> None:
    """
    Create a stacked visualization with latent spaces, color slices, and loss.

    Args:
        resultset: Resultset with latent_cube, color_slice_cube, and loss_cube
        latent_dims: Tuple of two LatentD specifications for the two latent panels
        max_error: Optional maximum error for consistent color scaling
        latent_annotations: Optional annotations to add to latent space plots
        nbid: Notebook ID for filename prefix
    """
    with displayer_mpl(
        f'large-assets/ex-{nbid}-results-{tags_for_file(resultset.tags)}.png',
        alt_text="""Composite figure with two latent panels (top), a color slice (middle), and a loss chart (bottom).""",
    ) as show:
        show(
            lambda theme: draw_stacked_results(
                resultset.latent_cube,
                resultset.color_slice_cube,
                resultset.loss_cube,
                latent_dims=latent_dims,
                theme=theme,
                max_error=max_error,
                latent_annotations=[
                    ann(theme) if not isinstance(ann, ConicalAnnotation) else ann for ann in latent_annotations
                ],
            )
        )


def scatter_similarity_vs_error(
    cube: ColorCube,
    anchor_hsv: tuple[float, float, float],
    *,
    anchor_name: str,
    theme: Theme,
    power: float,
) -> Figure:
    """
    Create a scatter plot of similarity to anchor vs reconstruction error.

    Args:
        cube: ColorCube with 'similarity' and 'MSE' variables
        anchor_hsv: Anchor color in HSV (for documentation)
        anchor_name: Name of the anchor color for axis label
        theme: Visual theme to use
        power: Power used for similarity computation (for axis label)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    draw_cube_scatter(ax, cube, theme=theme, x_var='similarity', y_var='MSE')
    ax.set_ylabel(r'MSE')
    ax.set_xlabel(rf'$\text{{sim}}_\text{{{anchor_name}}}^{{{power:.2g}}}$')
    ax.legend(loc='upper left')
    return fig


def visualize_error_vs_similarity(
    cube: ColorCube,
    anchor_hsv: tuple[float, float, float],
    *,
    tags: Sequence[str] = (),
    anchor_name: str = 'anchor',
    power: float,
    nbid: str,
) -> None:
    """
    Visualize reconstruction error vs similarity to an anchor color.

    Args:
        cube: ColorCube with 'similarity' and 'MSE' variables
        anchor_hsv: Anchor color in HSV
        tags: Tags to include in filename
        anchor_name: Name of the anchor color
        power: Power used for similarity computation
        nbid: Notebook ID for filename prefix
    """
    with displayer_mpl(
        f'large-assets/ex-{nbid}-error-vs-similarity-{tags_for_file(tags)}.png',
        alt_text=f"""Scatter plot showing reconstruction error versus similarity to {anchor_name}. Each point represents a color, with its position on the x-axis indicating how similar it is to pure red, and its position on the y-axis indicating the reconstruction error (mean squared error) for that color. The points are colored according to their actual color values.""",
    ) as show:
        show(
            lambda theme: scatter_similarity_vs_error(
                cube,
                anchor_hsv,
                theme=theme,
                anchor_name=anchor_name,
                power=power,
            )
        )


def hstack_named_results(*resultsets: Any) -> Any:  # pd.DataFrame
    """
    Create a table comparing results across multiple experiments.

    Args:
        *resultsets: Variable number of Resultset objects

    Returns:
        DataFrame with MSE columns for each resultset and delta columns
    """
    names = [' '.join(r.tags) for r in resultsets]
    df = resultsets[0].named_colors[['name', 'rgb', 'hsv', 'MSE']].rename(columns={'MSE': names[0]})
    for name, r in zip(names[1:], resultsets[1:], strict=True):
        df = df.merge(
            r.named_colors[['name', 'MSE']].rename(columns={'MSE': name}),
            on='name',
        )
        df[f'{name}-delta'] = df[name] - df[names[0]]
    return df


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


def show_dopesheet_as_md(dopesheet: Dopesheet, nbid: str) -> None:
    """
    Display a parameter schedule dopesheet with markdown and a plot.

    Args:
        dopesheet: Dopesheet object with parameter schedule
        title: Title for the plot
        nbid: Notebook ID for filename prefix
    """
    from IPython.display import Markdown, display

    display(Markdown(f"""## Parameter schedule \n{dopesheet.to_markdown()}"""))


def visualize_dopesheet(dopesheet: Dopesheet, nbid: str) -> None:
    """
    Display a parameter schedule dopesheet with markdown and a plot.

    Args:
        dopesheet: Dopesheet object with parameter schedule
        nbid: Notebook ID for filename prefix
    """
    with displayer_mpl(
        f'large-assets/ex-{nbid}-dopesheet.png',
        alt_text="""Plot showing the parameter schedule for the training run. The plot has two sections: the upper section shows various regularization weights over time, and the lower section shows the learning rate over time. The x-axis represents training steps.""",
    ) as show:
        show(lambda theme: plot_dopesheet(dopesheet, theme))


class NbViz:
    """
    Helper class to manage visualization state and utilities.

    Attributes:
        nbid: Notebook ID for filename prefix
    """

    def __init__(self, nbid: str) -> None:
        self.nbid = nbid

    def vis_recon_loss(self, data: ColorCube, tags: Sequence[str] = ()):
        visualize_reconstruction_loss(data, tags=tags, nbid=self.nbid)

    def vis_recon_colors(self, data: ColorCube, tags: Sequence[str] = ()):
        visualize_reconstructed_cube(data, tags=tags, nbid=self.nbid)

    def vis_stacked(
        self,
        resultset: Resultset,
        latent_dims: tuple[LatentD, LatentD],
        max_error: float | None = None,
        latent_annotations: Sequence[ConicalAnnotation | Callable[[Theme], ConicalAnnotation]] = (),
    ):
        visualize_stacked_results(
            resultset,
            latent_dims=latent_dims,
            max_error=max_error,
            latent_annotations=latent_annotations,
            nbid=self.nbid,
        )

    def vis_error_vs_similarity(
        self,
        cube: ColorCube,
        anchor_hsv: tuple[float, float, float],
        tags: Sequence[str] = (),
        anchor_name: str = 'anchor',
        power: float = 2.0,
    ):
        visualize_error_vs_similarity(
            cube,
            anchor_hsv,
            tags=tags,
            anchor_name=anchor_name,
            power=power,
            nbid=self.nbid,
        )

    def dopesheet_md(self, dopesheet: Dopesheet):
        show_dopesheet_as_md(dopesheet, nbid=self.nbid)

    def vis_dopesheet(self, dopesheet: Dopesheet):
        visualize_dopesheet(dopesheet, nbid=self.nbid)
