from typing import Callable, Collection, Sequence

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from ex_color.data import ColorCube
from ex_color.evaluation import TestSet, hstack_named_results, error_correlation
from ex_color.vis import (
    ColorTableHtmlFormatter,
    ColorTableLatexFormatter,
    ConicalAnnotation,
    draw_stacked_results,
    plot_colors,
    plot_cube_series,
    plot_latent_grid_3d_from_cube,
)
from ex_color.vis.plot_dopesheet import plot_dopesheet
from ex_color.vis.plot_latent_slices import LatentD
from ex_color.vis.plot_similarity import scatter_similarity_vs_error
from mini.temporal.dopesheet import Dopesheet
from utils.nb import displayer_mpl
from utils.plt import Theme
from utils.strings import sup


class NbViz:
    def __init__(self, nbid: str):
        self.nbid = nbid

    def tab_dopesheet(self, dopesheet: Dopesheet):
        display(Markdown(f"""## Parameter schedule \n{dopesheet.to_markdown()}"""))

    def plot_dopesheet(self, dopesheet: Dopesheet):
        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-dopesheet.png',
            alt_text="""Plot showing the parameter schedule for the training run, titled "{title}". The plot has two sections: the upper section shows various regularization weights over time, and the lower section shows the learning rate over time. The x-axis represents training steps.""",
        ) as show:
            show(lambda theme: plot_dopesheet(dopesheet, theme))

    def plot_cube(
        self,
        data: ColorCube | TestSet,
        *,
        tags: Collection[str] | None = None,
        colors: str = 'recon',
        colors_compare: str = 'color',
    ):
        tag_list = list(tags) if tags else list(data.tags) if isinstance(data, TestSet) else []
        data = data if isinstance(data, ColorCube) else data.color_slice_cube.permute('svh')

        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-pred-colors-{tags_for_file(tag_list)}.png',
            alt_text="""Plot showing four slices of the HSV cube, titled "{title}". Nominally, each slice has constant saturation, but varies in value (brightness) from top to bottom, and in hue from left to right. Each color value is represented as a square patch of that color. The outer portion of the patches shows the color as reconstructed by the model; the inner portion shows the true (input) color.""",
        ) as show:
            show(
                lambda: plot_colors(
                    data,
                    title=f'Predicted colors · {" · ".join(tag_list)}',
                    colors=colors,
                    colors_compare=colors_compare,
                )
            )

    def plot_recon_loss(
        self, data: ColorCube | TestSet, *, tags: Collection[str] | None = None, error_var: str = 'MSE'
    ):
        tag_list = list(tags) if tags else list(data.tags) if isinstance(data, TestSet) else []
        data = data if isinstance(data, ColorCube) else data.loss_cube

        max_loss = np.max(data[error_var])
        median_loss = np.median(data[error_var])
        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-loss-colors-{tags_for_file(tag_list)}.png',
            alt_text=f"""Line chart showing loss per color, titled "{{title}}". Y-axis: mean square error, ranging from zero to {max_loss:.2g}. X-axis: hue.""",
        ) as show:
            show(
                lambda: plot_cube_series(
                    data.permute('hsv')[:, -1:, :: (data.shape[2] // -5)],
                    data.permute('svh')[:, -1:, :: -(data.shape[0] // -3)],
                    data.permute('vsh')[:, -1:, :: -(data.shape[0] // -3)],
                    title=f'Reconstruction error · {" · ".join(tag_list)}',
                    var=error_var,
                    figsize=(12, 3),
                )
            )
        print(f'Max loss: {max_loss:.2g}')
        print(f'Median MSE: {median_loss:.2g}')

    def plot_latent_space(
        self,
        data: ColorCube | TestSet,
        *,
        tags: Collection[str] | None = None,
        dims: Sequence[tuple[int, int, int]],
        colors: str = 'recon',
        colors_compare: str = 'color',
        latents: str = 'latents',
    ):
        tag_list = list(tags) if tags else list(data.tags) if isinstance(data, TestSet) else []
        data = data if isinstance(data, ColorCube) else data.latent_cube

        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-latents-{tags_for_file(tag_list)}.png',
            alt_text="""Two rows of three spherical plots, titled "{title}". Each plot shows a vibrant collection of colored circles or balls scattered over the surface of a hypersphere, with each plot showing one 2D projection.""",
        ) as show:
            show(
                lambda theme: plot_latent_grid_3d_from_cube(
                    data,
                    colors=colors,
                    colors_compare=colors_compare,
                    latents=latents,
                    title=f'Latents ·  · {" · ".join(tag_list)}',
                    dims=dims,
                    dot_radius=10,
                    theme=theme,
                )
            )

    def plot_stacked_results(
        self,
        res: TestSet,
        *,
        latent_dims: tuple[LatentD, LatentD],
        max_error: float | None = None,
        latent_annotations: Sequence[ConicalAnnotation | Callable[[Theme], ConicalAnnotation]] = (),
    ):
        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-results-{tags_for_file(list(res.tags))}.png',
            alt_text="""Composite figure with two latent panels (top), a color slice (middle), and a loss chart (bottom).""",
        ) as show:
            show(
                lambda theme: draw_stacked_results(
                    res.latent_cube,
                    res.color_slice_cube,
                    res.loss_cube,
                    latent_dims=latent_dims,
                    theme=theme,
                    max_error=max_error,
                    latent_annotations=[
                        ann(theme) if not isinstance(ann, ConicalAnnotation) else ann  #
                        for ann in latent_annotations
                    ],
                )
            )

    def plot_error_vs_similarity(
        self,
        data: ColorCube | TestSet,
        anchor_hsv: tuple[float, float, float],
        *,
        tags: Collection[str] | None = None,
        anchor_name: str = 'anchor',
        power: float,
    ):
        tag_list = list(tags) if tags else list(data.tags) if isinstance(data, TestSet) else []
        cube = data if isinstance(data, ColorCube) else data.latent_cube

        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-error-vs-similarity-{tags_for_file(tag_list)}.png',
            alt_text=f"""Scatter plot showing reconstruction error versus similarity to {anchor_name}. Each point represents a color, with its position on the x-axis indicating how similar it is to pure red, and its position on the y-axis indicating the reconstruction error (mean squared error) for that color. The points are colored according to their actual color values.""",
        ) as show:
            show(
                lambda theme: scatter_similarity_vs_error(
                    cube, anchor_hsv, theme=theme, anchor_name=anchor_name, power=power
                )
            )

        corr, p_value = error_correlation(cube, (0, 1, 1), power=power)
        print(f'MSE,sim{sup(power)} {",".join(tag_list)}: r = {corr:.2f}, R²: {corr**2:.2f}, p = {p_value:.3g}')

    def plot_boxplot(
        self,
        data: Sequence[float] | pd.Series,
        *,
        figsize: tuple[int, int] = (4, 1),
        ylabel: str,
        tags: Collection[str] | None = None,
        xlim: tuple[float | None, float | None] | None = None,
        log_scale: bool = False,
    ):
        """
        Create a horizontal box plot for a single variable.

        Args:
            data: Sequence of values to plot.
            figsize: Figure size as (width, height).
            ylabel: Label for the y-axis (typically a LaTeX-formatted metric name).
            tags: Optional tags to include in the filename.
            xlim: Tuple of (min, max) for x-axis limits. Use None for auto.
            log_scale: Whether to use a logarithmic scale for the x-axis.
        """
        tag_list = list(tags) if tags else []

        with displayer_mpl(
            f'large-assets/ex-{self.nbid}-boxplot-{tags_for_file(tag_list)}.png',
            alt_text=f"""Horizontal box plot showing the distribution of {ylabel}.""",
        ) as show:
            show(lambda: draw_boxplot(data, figsize=figsize, ylabel=ylabel, xlim=xlim, log_scale=log_scale))

    def tab_error_vs_color(self, *res: TestSet):
        df = hstack_named_results(*res)
        display(ColorTableHtmlFormatter().style(df))

    def tab_error_vs_color_latex(self, baseline: TestSet, *res: TestSet):
        """
        Create a LaTeX table of results across several experiments.

        Args:
            baseline: The baseline test set to compare against.
            *res: The result test sets to include in the table.
        """
        df = hstack_named_results(baseline, *res)

        formatter = ColorTableLatexFormatter()
        # print(formatter.preamble)
        delta_cols = [col for col in df.columns if col.endswith('-delta')]
        latex = formatter.to_str(
            pd.DataFrame(
                {
                    'color': df['name'].str.capitalize(),
                    'rgb': df['rgb'],
                    'baseline': df['baseline'],
                    **{col.rstrip('-delta'): df[col] for col in delta_cols},
                }
            ),
            caption='Reconstruction error by color and intervention method',
            label='tab:placeholder',
        )
        display({'text/markdown': f'```latex\n{latex}\n```', 'text/plain': latex}, raw=True)


def draw_boxplot(
    data: Sequence[float] | pd.Series,
    *,
    figsize: tuple[int, int] = (4, 1),
    ylabel: str,
    xlim: tuple[float | None, float | None] | None = None,
    log_scale: bool = False,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, vert=False, widths=[figsize[0] / 16])
    if not ylabel:
        ax.set_yticks([])
    else:
        ax.set_yticklabels([ylabel])
    if log_scale:
        ax.set_xscale('log')
    # else:
    #     ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    return fig


class ThemedAnnotation:
    def __init__(self, direction: Sequence[float], angle: float, dashed: bool = False):
        self.direction = direction
        self.angle = angle
        self.dashed = dashed

    def __call__(self, theme: Theme) -> ConicalAnnotation:
        return ConicalAnnotation(
            direction=self.direction,
            angle=self.angle,
            color=theme.val('black', dark='#fff'),
            linewidth=theme.val(0.75, dark=1),
            **dict(
                dashes=theme.val((8, 8), dark=(4, 4)),
                gapcolor=theme.val('#ddda', dark='#222a'),
            ) if self.dashed else {}
        )  # fmt: skip


def tags_for_file(tags: Sequence[str]) -> str:
    import re

    tags = [re.sub(r'[^a-zA-Z0-9]+', '-', tag.lower()) for tag in tags]
    return '-'.join(tags)
