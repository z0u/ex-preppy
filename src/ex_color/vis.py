from ex_color.data.color_cube import ColorCube


def plot_colors(
    cube: ColorCube,
    pretty: bool | str = True,
    patch_size: float = 0.25,
    title: str = '',
):
    """Plot a ColorCube in 2D slices."""
    from math import ceil
    import matplotlib.pyplot as plt
    from itertools import chain

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

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = list(chain(*axes))  # Flatten the axes array

    # Plot each slice of the cube (one for each value)
    for i, ax in enumerate(axes):
        if i >= len(main_coords):
            ax.set_visible(False)
            continue
        row = i // n_cols
        col = i % n_cols

        ax.imshow(cube.rgb_grid[i])

        ax.set_title(f'{main_axis} = {fmt(main_axis, main_coords[i])}', fontsize=8)

        # Add axes labels without cluttering the display
        if row == n_rows - 1:
            ax.xaxis.set_ticks([0, len(x_coords) - 1])
            coord1 = fmt(x_axis, x_coords[0])
            coord2 = fmt(x_axis, x_coords[-1])
            ax.xaxis.set_ticklabels([coord1, coord2])
            ax.xaxis.set_tick_params(labelsize=8)
            ax.set_xlabel(x_axis.upper(), fontsize=8)
        else:
            ax.xaxis.set_visible(False)

        if col == 0:
            ax.yaxis.set_ticks([0, len(y_coords) - 1])
            coord1 = fmt(y_axis, y_coords[0])
            coord2 = fmt(y_axis, y_coords[-1])
            ax.yaxis.set_ticklabels([coord1, coord2])
            ax.yaxis.set_tick_params(labelsize=8)
            ax.set_ylabel(y_axis.upper(), fontsize=8)
        else:
            ax.yaxis.set_visible(False)

    _title = f'{title} - ' if title else ''
    plt.suptitle(
        f'{_title}{cube.canonical_space.upper()} as {x_axis.upper()},{y_axis.upper()} per {main_axis.upper()}',
        color='gray',
    )

    # Light and dark mode compatibility = compromise on both!
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)
        ax.title.set_color('gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        ax.tick_params(colors='gray')

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
