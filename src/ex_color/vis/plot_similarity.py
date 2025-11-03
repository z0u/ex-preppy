import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from ex_color.data import ColorCube, hsv_similarity
from ex_color.vis import (
    draw_cube_scatter,
)
from utils.plt import Theme


def scatter_similarity_vs_error(
    cube: ColorCube,
    anchor_hsv: tuple[float, float, float],
    *,
    anchor_name: str,
    theme: Theme,
    power: float,
):
    """Scatter plot of similarity to anchor vs reconstruction error."""
    cube = cube.assign(hsv=ski.color.rgb2hsv(cube['color']))
    cube = cube.assign(similarity=hsv_similarity(cube['hsv'], np.array(anchor_hsv), hemi=True, mode='angular') ** power)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    draw_cube_scatter(ax, cube, theme=theme, x_var='similarity', y_var='MSE')
    ax.set_ylabel(r'MSE')
    ax.set_xlabel(rf'$\text{{sim}}_\text{{{anchor_name}}}^{{{power:.2g}}}$')
    ax.legend(loc='upper left')
    return fig


def scatter_vibrancy_vs_error(
    cube: ColorCube,
    *,
    theme: Theme,
    power: float,
):
    """Scatter plot of vibrancy (saturation * value) vs reconstruction error."""
    cube = cube.assign(hsv=ski.color.rgb2hsv(cube['color']))
    # Vibrancy is saturation * value
    vibrancy = cube['hsv'][..., 1] * cube['hsv'][..., 2]
    cube = cube.assign(vibrancy=vibrancy**power)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    draw_cube_scatter(ax, cube, theme=theme, x_var='vibrancy', y_var='MSE')
    ax.set_ylabel(r'MSE')
    ax.set_xlabel(rf'$\text{{vibrancy}}^{{{power:.2g}}}$')
    ax.legend(loc='upper left')
    return fig
