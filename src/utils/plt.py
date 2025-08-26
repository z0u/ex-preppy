import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.mplot3d import Axes3D


def hide_decorations(
    ax: Axes | PolarAxes | Axes3D, background: bool = True, ticks: bool = True, border: bool = True, grid: bool = True
) -> None:
    """Remove all decorations from the axes."""
    if background:
        ax.patch.set_alpha(0)
        if isinstance(ax, Axes3D):
            ax.xaxis.set_pane_color('none')
            ax.yaxis.set_pane_color('none')
            ax.zaxis.set_pane_color('none')

    if ticks:
        if isinstance(ax, PolarAxes):
            ax.set_rticks([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    if border:
        if isinstance(ax, PolarAxes):
            ax.spines['polar'].set_visible(False)
        elif isinstance(ax, Axes3D):
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


def configure_matplotlib():
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#222'
    plt.rcParams['figure.facecolor'] = '#333'
    plt.rcParams['figure.dpi'] = 150

    # Hide spines (borders)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False
    # Make axis tick font smaller
    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'
    # Make ticks semi-opaque
    plt.rcParams['xtick.color'] = '#fff8'
    plt.rcParams['xtick.labelcolor'] = '#fff'
    plt.rcParams['ytick.color'] = '#fff8'
    plt.rcParams['ytick.labelcolor'] = '#fff'
