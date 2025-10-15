from ex_color.vis.plot_color_loss import draw_cube_series_on_ax, plot_cube_series
from ex_color.vis.plot_cube import draw_color_slice, plot_colors
from ex_color.vis.plot_cube_scatter import draw_cube_scatter, plot_cube_scatter
from ex_color.vis.plot_latent_slices import (
    ConicalAnnotation,
    draw_latent_panel,
    draw_latent_panel_from_cube,
    plot_latent_grid_3d,
    plot_latent_grid_3d_from_cube,
)
from ex_color.vis.stacked_fig import StackedFigure, build_stacked_figure, draw_stacked_results
from ex_color.vis.tabular import ColorTableHtmlFormatter, ColorTableLatexFormatter
from ex_color.vis.helpers import NbViz

__all__ = [
    'build_stacked_figure',
    'ColorTableHtmlFormatter',
    'ColorTableLatexFormatter',
    'NbViz',
    'ConicalAnnotation',
    'draw_color_slice',
    'draw_cube_series_on_ax',
    'draw_cube_scatter',
    'draw_latent_panel_from_cube',
    'draw_latent_panel',
    'draw_stacked_results',
    'plot_colors',
    'plot_cube_series',
    'plot_cube_scatter',
    'plot_latent_grid_3d_from_cube',
    'plot_latent_grid_3d',
    'StackedFigure',
]
