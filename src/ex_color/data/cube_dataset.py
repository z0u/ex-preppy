from typing import Callable, Literal, overload

import numpy as np
import skimage as ski
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from ex_color.data.color_cube import ColorCube


# Updated labelling functions to work with numpy arrays
def redness(rgb_array: np.ndarray) -> np.ndarray:
    """Compute redness for numpy RGB array with shape [..., 3]"""
    r, g, b = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    return r * (1 - g / 2 - b / 2)


def greenness(rgb_array: np.ndarray) -> np.ndarray:
    """Compute greenness for numpy RGB array with shape [..., 3]"""
    r, g, b = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    return g * (1 - r / 2 - b / 2)


def blueness(rgb_array: np.ndarray) -> np.ndarray:
    """Compute blueness for numpy RGB array with shape [..., 3]"""
    r, g, b = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    return b * (1 - r / 2 - g / 2)


def vibrancy(rgb_array: np.ndarray) -> np.ndarray:
    """Compute vibrancy (saturation) for numpy RGB array with shape [..., 3]"""
    grid = ski.color.rgb2hsv(rgb_array)

    S_grid = grid[..., 1]
    V_grid = grid[..., 2]

    # Vibrant focus = S * V (ranges 0-1)
    return S_grid * V_grid


class CubeDataset(Dataset[dict[str, Tensor]]):
    """PyTorch Dataset that wraps a ColorCube."""

    def __init__(self, cube: ColorCube):
        self.cube = cube
        self.length = int(np.prod(cube.shape))

        # Flatten all variables and convert to tensors for easy indexing
        self.flat_vars = {}
        for name, var in cube.vars.items():
            if name == 'color':
                # RGB data has shape [..., 3]
                flattened = var.reshape(-1, 3)
                self.flat_vars[name] = torch.tensor(flattened, dtype=torch.float32)
            else:
                # Other variables have shape [...]
                flattened = var.flatten()
                self.flat_vars[name] = torch.tensor(flattened, dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return a single sample with all variables as tensors."""
        return {name: var[idx] for name, var in self.flat_vars.items()}


def stochastic_labels(batch) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Collate function that discretizes precomputed label probabilities.

    Args:
        batch: List of dicts from CubeDataset, each containing tensors

    Returns:
        Tuple of (colors, discretized_labels)
    """
    # Extract colors (already tensors from CubeDataset)
    colors = torch.stack([item['color'] for item in batch])

    # Extract and discretize label probabilities
    labels = {}
    label_names = [key for key in batch[0].keys() if key not in ['color', 'bias']]

    for label_name in label_names:
        probs = torch.stack([item[label_name] for item in batch])
        # Stochastically discretize
        rand = torch.rand_like(probs)
        labels[label_name] = (rand < probs).float()

    return colors, labels


def exact_labels(batch) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Collate function that selects labels with probability 1.

    Args:
        batch: List of dicts from CubeDataset, each containing tensors

    Returns:
        Tuple of (colors, discretized_labels)
    """
    # Extract colors (already tensors from CubeDataset)
    colors = torch.stack([item['color'] for item in batch])

    # Extract raw label probabilities
    labels = {
        key: torch.stack([item[key] for item in batch])  #
        for key in batch[0].keys()
        if key not in ['color', 'bias']
    }

    return colors, labels


@overload
def prep_color_dataset(
    subs: int,
    sample_at: Literal['cell-corners'],
    **attributes: Callable[[np.ndarray], np.ndarray],
) -> CubeDataset: ...
@overload
def prep_color_dataset(
    subs: int,
    sample_at: Literal['cell-centers'],
    *,
    add_corners: bool = True,
    **attributes: Callable[[np.ndarray], np.ndarray],
) -> ConcatDataset[dict[str, Tensor]]: ...
def prep_color_dataset(
    subs: int,
    sample_at='cell-corners',
    *,
    add_corners=None,
    **attributes: Callable[[np.ndarray], np.ndarray],
):
    """
    Prepare an RGB cube dataset.

    Args:
        subs: Number of subdivisions along each axis
        sample_at: 'cell-corners' or 'cell-centers' to specify sampling strategy
        add_corners: Whether to add corner points (only relevant if sampling at cell centers)
        **attributes: Additional attributes to compute from RGB colors

    If sampling at cell corners, the dataset will contain `subs³` points. If
    sampling at cell centers, the dataset will contain `(subs-1)³` points plus
    optionally the `8` corner points if `add_corners` is True (default).
    """
    if sample_at == 'cell-centers':
        # Generate points at cell centers
        offset = 1 / (2 * subs)
        coords = np.linspace(offset, 1 - offset, subs - 1)
    else:
        # Generate points at cell corners
        coords = np.linspace(0, 1, subs)
    cube = ColorCube.from_rgb(r=coords, g=coords, b=coords)
    for name, func in attributes.items():
        cube = cube.assign(name, func(cube['color']))

    if add_corners is None:
        add_corners = sample_at == 'cell-centers'  # Default to add corners if offset is used
    if add_corners:
        # Add the corners of the cube as key points - useful for validation
        # data, where the points are the centers of the cells but we still want
        # to check the primary and secondary colors, and black and white.
        corner_coords = np.array([0.0, 1.0])
        corner_cube = ColorCube.from_rgb(r=corner_coords, g=corner_coords, b=corner_coords)
        for name, func in attributes.items():
            corner_cube = corner_cube.assign(name, func(corner_cube['color']))
        return ConcatDataset([CubeDataset(cube), CubeDataset(corner_cube)])

    return CubeDataset(cube)
