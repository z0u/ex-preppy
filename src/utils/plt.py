import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def hide_decorations(ax: Axes, background: bool = True, ticks: bool = True, border: bool = True) -> None:
    """Remove all decorations from the axes."""
    if background:
        ax.patch.set_alpha(0)
    if ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if border:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


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
