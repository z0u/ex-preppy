import numpy as np
from ex_color.data.color_cube import ColorCube
from ex_color.data.grid import coordinate_grid


def vibrancy(cube: ColorCube) -> np.ndarray:
    """
    Create a weights array focusing on vibrant colors (high S, high V).

    Args:
        cube: An HSV ColorCube instance to create the vibrant focus for.

    Returns:
        An array with the same shape as the cube, with ones for vibrant colors
        (high S and high V), zeros for black (V=0) and grays (S=0), and
        intermediate values for other colors.
    """
    if cube.canonical_space != 'hsv':
        raise ValueError(f'Cannot create vibrant focus for non-HSV cube ({cube.canonical_space}).')
    if 'v' not in cube.space or 's' not in cube.space:
        raise ValueError(f'Cannot create vibrant focus for cube with space {cube.space}. Must include "s" and "v".')

    # Need the original grid shape to find S and V
    grid = coordinate_grid(*cube.coordinates)

    s_idx = cube.space.index('s')
    v_idx = cube.space.index('v')
    S_grid = grid[..., s_idx]
    V_grid = grid[..., v_idx]

    # Vibrant focus = S * V (ranges 0-1)
    vibrant_focus = S_grid * V_grid
    return vibrant_focus


class CubeRandomSampler:
    """Samples points from a ColorCube based on its intrinsic bias, modulated by externally provided focus weights."""

    _effective_bias: np.ndarray | None
    """The normalized bias used for sampling, or None if no points are available."""

    def __init__(self, cube: ColorCube):
        """
        Create a sampler for a ColorCube.

        Args:
            cube: The ColorCube instance to sample from.
        """
        self.cube = cube
        self.original_shape = cube.rgb_grid.shape[:-1]
        self.n_points = np.prod(self.original_shape)
        self.rng = np.random.default_rng()

        # Flatten arrays for efficient sampling
        self.flat_rgb = cube.rgb_grid.reshape(self.n_points, 3)

        # Initialize focus weights to uniform (allow all points initially)
        self.set_focus_weights(np.ones_like(cube.bias))

    def set_focus_weights(self, weights: np.ndarray):
        """
        Set the focus weights used to modulate the base bias.

        Args:
            weights: An array with the same shape as the original cube grid
                (self.original_shape), representing the desired focus. Values
                should ideally be non-negative. There must be at least one
                non-zero value in the weights array to allow sampling.
        """
        effective_bias = np.maximum(weights * self.cube.bias, 0.0)

        bias_sum = np.sum(effective_bias)
        if bias_sum > 1e-9:
            self._effective_bias = (effective_bias / bias_sum).flatten()
        else:
            # If the sum is effectively zero, we cannot sample.
            self._effective_bias = None

    def sample(self, k: int) -> np.ndarray:
        """
        Draw k samples from the cube based on the current effective bias.

        Args:
            k: The number of samples to draw.

        Returns:
            An array of shape (k, 3) containing the sampled RGB values.

        Raises:
            ValueError: If no points are available for sampling with the current
                        bias and focus settings (i.e., effective bias sum is zero).
        """
        if self._effective_bias is None:
            raise ValueError('No points available for sampling.')

        # np.random.choice requires probabilities sum to 1.
        sampled_indices = self.rng.choice(
            self.n_points,
            size=k,
            replace=True,  # Allow sampling the same point multiple times
            p=self._effective_bias,
        )
        # Return the RGB values corresponding to the sampled indices
        return self.flat_rgb[sampled_indices]
