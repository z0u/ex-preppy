import logging
from typing import Iterator, List, TypeAlias

from skimage.color import rgb2hsv
import numpy as np
import numpy.typing as npt
from torch.utils.data import Sampler

from ex_color.data.color_cube import ColorCube
from ex_color.data.grid import coordinate_grid

log = logging.getLogger(__name__)


def primary_secondary_focus(cube: ColorCube) -> npt.NDArray[np.float64]:
    """
    Create a weights array focusing on primary and secondary colors.

    Weights are 1.0 for hues corresponding to primary (0, 120, 240 deg) and
    secondary (60, 180, 300 deg) colors, falling off to 0.0 for tertiary
    hues (halfway between primary/secondary). The calculation is based purely
    on hue, ignoring saturation and value.

    Args:
        cube: An HSV ColorCube instance to create the focus weights for.

    Returns:
        An array with the same shape as the cube, with weights based on hue proximity
        to primary/secondary colors.
    """
    if cube.canonical_space != 'hsv':
        raise ValueError(f'Cannot create primary/secondary focus for non-HSV cube ({cube.canonical_space}).')

    # Need the original grid shape to find H
    grid = coordinate_grid(*cube.coordinates)

    h_idx = cube.space.index('h')
    H_grid = grid[..., h_idx]  # Hue is normalized 0-1

    # Convert normalized hue (0-1) to degrees (0-360)
    H_degrees = H_grid * 360.0

    # Calculate the angle modulo 60 degrees. Primary/secondary hues are at multiples of 60.
    angle_mod_60 = H_degrees % 60.0

    # Find the minimum angular distance to the nearest multiple of 60 degrees.
    # This distance ranges from 0 (at a primary/secondary) to 30 (at a tertiary).
    dist_to_nearest_60 = np.minimum(angle_mod_60, 60.0 - angle_mod_60)

    # Map the distance (0-30) to a weight (1.0-0.0) using a cosine function.
    # cos(0) = 1, cos(pi) = -1. We scale the distance to range 0 to pi.
    # (cos(scaled_dist) + 1) / 2 maps [0, pi] -> [1, 0]
    weights = (np.cos(dist_to_nearest_60 * np.pi / 30.0) + 1.0) / 2.0

    return weights


def vibrancy(cube: ColorCube) -> npt.NDArray[np.float64]:
    """
    Create a weights array focusing on vibrant colors (high S, high V).

    Args:
        cube: A ColorCube instance to create the vibrant focus for.

    Returns:
        An array with the same shape as the cube, with ones for vibrant colors
        (high S and high V), zeros for black (V=0) and grays (S=0), and
        intermediate values for other colors.
    """
    grid = rgb2hsv(cube.rgb_grid)

    S_grid = grid[..., 1]
    V_grid = grid[..., 2]

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
        self.original_shape = cube.shape  # Use the new shape property
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


# S = TypeVar('S', bound=tuple[int, ...])
# Weights: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]
Weights: TypeAlias = npt.NDArray[np.floating]


class DynamicWeightedRandomBatchSampler(Sampler[List[int]]):
    """
    Samples batches of indices from a ColorCube.

    Samples based on the cube's intrinsic bias, modulated by externally provided focus
    weights that can be updated dynamically.

    Designed to be used as a `batch_sampler` in `torch.utils.data.DataLoader`.
    """

    bias: Weights
    batch_size: int
    steps_per_epoch: int

    _weights: Weights
    _effective_bias: Weights | None
    """The normalized bias used for sampling, or None if no points are available."""
    _seed: int | None
    _rng: np.random.Generator  # Use a specific Generator instance

    def __init__(
        self,
        bias: Weights,
        batch_size: int,
        steps_per_epoch: int,
        weights: Weights | None = None,
        seed: int | None = None,
    ):
        """
        Create a batch sampler for a ColorCube.

        Args:
            bias: The base bias weights for sampling.
            batch_size: The number of indices to yield in each batch.
            steps_per_epoch: The total number of batches to yield per iteration (epoch).
            weights: An optional array with the same shape as the bias, used to modulate the sampling bias.
            seed: An optional integer seed for the random number generator.

        Set the focus weights using the `weights` property.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f'batch_size should be a positive integer value, but got batch_size={batch_size}')
        if not isinstance(steps_per_epoch, int) or steps_per_epoch < 0:
            raise ValueError(
                f'steps_per_epoch should be a non-negative integer value, but got steps_per_epoch={steps_per_epoch}'
            )

        self.bias = bias
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.seed = seed
        self.weights = weights if weights is not None else np.ones_like(bias)

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter
    def seed(self, seed: int | None):
        """Reset the random number generator with the given seed."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def weights(self) -> Weights:
        """
        The focus weights used to modulate the base bias.

        Must be the same shape as self.bias.

        The effective bias for the next sampled batch will be based on these weights.
        """
        return self._weights

    @weights.setter
    def weights(self, weights: Weights):
        # Combine external weights with the cube's intrinsic bias
        effective_bias = np.maximum(weights * self.bias, 0.0)

        bias_sum: float = np.sum(effective_bias)
        if bias_sum > 1e-9:
            # Normalize and flatten for sampling (choice requires 1D array)
            self._effective_bias = (effective_bias / bias_sum).flatten()
        else:
            # If the sum is effectively zero, we cannot sample.
            self._effective_bias = None
        self._weights = weights

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices, calculated using the latest focus weights."""
        warned = False

        for _ in range(self.steps_per_epoch):
            if self._effective_bias is None:
                if not warned:
                    log.warning('Sampling weights are undefined. Using uniform sampling.')
                    warned = True

            yield self._rng.choice(
                np.prod(self.bias.size),
                size=self.batch_size,
                replace=True,
                p=self._effective_bias,
            ).tolist()  # type: ignore[no-untyped-call]

    def __len__(self) -> int:
        """Return the number of batches that will be yielded."""
        return self.steps_per_epoch
