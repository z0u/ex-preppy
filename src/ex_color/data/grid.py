from typing import TypeAlias

import numpy as np
import numpy.typing as npt

CoordinateAxis: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]


def coordinate_grid(*coordinates: CoordinateAxis) -> npt.NDArray[np.floating]:
    """
    Create a coordinate grid for n-dimensional space.

    Args:
        coordinates: List of 1D arrays representing the coordinates along each axis.

    Returns:
        An n+1D array where the last dimension contains the indices for each coordinate.
    """
    # Create meshgrid
    grids = np.meshgrid(*coordinates, indexing='ij')

    # Stack along a new last dimension
    return np.stack(grids, axis=-1)
