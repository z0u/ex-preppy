from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axis3d as axis3d
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.projections import PolarAxes
from mpl_toolkits.mplot3d import Axes3D


def hide_decorations(
    ax: Axes | PolarAxes | Axes3D, background: bool = True, ticks: bool = True, border: bool = True, grid: bool = True
) -> None:
    """Remove all decorations from the axes."""
    if background:
        ax.patch.set_alpha(0)
        if isinstance(ax, Axes3D):
            assert isinstance(ax.xaxis, axis3d.Axis)
            assert isinstance(ax.yaxis, axis3d.Axis)
            assert isinstance(ax.zaxis, axis3d.Axis)
            ax.xaxis.set_pane_color('none')
            ax.yaxis.set_pane_color('none')
            ax.zaxis.set_pane_color('none')

    if ticks:
        if isinstance(ax, PolarAxes):
            ax.set_rticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])  # type:ignore

    if border:
        if isinstance(ax, PolarAxes):
            ax.spines['polar'].set_visible(False)
        elif isinstance(ax, Axes3D):
            assert isinstance(ax.xaxis, axis3d.Axis)
            assert isinstance(ax.yaxis, axis3d.Axis)
            assert isinstance(ax.zaxis, axis3d.Axis)
            ax.xaxis.line.set_color('none')
            ax.yaxis.line.set_color('none')
            ax.zaxis.line.set_color('none')
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    if grid:
        ax.grid(False)


Theme = Literal['base', 'light', 'dark', 'transparent'] | Mapping[str, str]
ThemeType = Literal['light', 'dark', 'indeterminate']


@contextmanager
def use_theme(*themes: Theme):
    with mpl.rc_context():
        stylesheet_dir = Path(__file__).parent / 'mplstyles'
        for theme in themes:
            if isinstance(theme, Mapping):
                plt.style.use(dict(theme))
            else:
                plt.style.use(stylesheet_dir / f'{theme}.mplstyle')
        yield


def autoclose(factory: Callable[..., Figure | None]) -> Callable[..., Figure | None]:
    @wraps(factory)
    def _autoclose(*args, **kwargs) -> Figure | None:
        fig = factory(*args, **kwargs)
        plt.close(fig)
        return fig

    return _autoclose
