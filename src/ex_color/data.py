import numpy as np


class ColorCube:
    """A tensor of RGB values with coordinates in various color spaces (e.g. HSV)."""

    coordinates: tuple[np.ndarray, ...]
    """The coordinate axes of the cube, with shape (a, b, c) for each axis."""
    space: str
    """The color space of the cube (e.g. 'vsh'), which is some permutation of the canonical space."""
    canonical_space: str
    """The well-known color space for the cube (e.g. 'hsv'), regardless of the current axis order."""
    rgb_grid: np.ndarray
    """The RGB values in the cube, with shape (a, b, c, 3)."""

    def __init__(
        self,
        rgb_grid: np.ndarray,
        coordinates: tuple[np.ndarray, ...],
        space: str,
        canonical_space: str,
    ):
        self.space = space
        self.canonical_space = canonical_space
        self.coordinates = coordinates
        self.rgb_grid = rgb_grid

    @classmethod
    def from_hsv(cls, h: np.ndarray, s: np.ndarray, v: np.ndarray):
        import skimage as ski

        grid = ski.color.hsv2rgb(coordinate_grid(h, s, v))
        return cls(grid, (h, s, v), 'hsv', 'hsv')

    @classmethod
    def from_rgb(cls, r: np.ndarray, g: np.ndarray, b: np.ndarray):
        grid = coordinate_grid(r, g, b)
        return cls(grid, (r, g, b), 'rgb', 'rgb')

    def permute(self, new_space: str):
        if set(self.space) != set(new_space):
            raise ValueError(f'Cannot permute {self.space} to {new_space}: different axes')
        indices = tuple(self.space.index(axis) for axis in new_space)
        new_grid = np.transpose(self.rgb_grid, indices + (-1,))
        new_coordinates = tuple(self.coordinates[i] for i in indices)
        return ColorCube(new_grid, new_coordinates, new_space, self.canonical_space)


def coordinate_grid(*coordinates: list | np.ndarray) -> np.ndarray:
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


def isclose_cyclic(
    a: float | np.floating | np.typing.NDArray[np.floating],
    b: float | np.floating | np.typing.NDArray[np.floating],
    atol: float | np.floating | np.typing.NDArray[np.floating] = 1e-8,
    period: float = 1.0,
) -> np.typing.NDArray[np.bool]:
    """Check if two arrays are close in a cyclic manner (e.g., angles)."""
    return (
        np.isclose(a, b, atol=atol) |
        np.isclose(a + period, b, atol=atol) |
        np.isclose(a - period, b, atol=atol)
    )  # fmt: skip


def isbetween_cyclic(
    a: float | np.floating | np.typing.NDArray[np.floating],
    lower: float,
    upper: float,
    period: float = 1.0,
    atol: float = 1e-8,
) -> np.typing.NDArray[np.bool]:
    """Check if a is between lower and upper in a cyclic manner (e.g., angles)."""
    _a = a % period
    lower = lower % period
    upper = upper % period
    if lower <= upper:
        return np.greater_equal(_a, lower - atol) & np.less_equal(_a, upper + atol)
    else:
        return ~(np.greater(_a, upper) & np.less(_a, lower))


def hue_arange(
    lower: float = 0.0,
    upper: float | None = None,
    step_size: float = 1 / 12,
    inclusive: bool = False,
) -> np.ndarray:
    """Generate an array of hue values."""
    lower = lower % 1.0
    if upper is None:
        upper = lower + 1.0
    else:
        upper = upper % 1.0
    if lower > upper:
        upper += 1.0
    if inclusive:
        upper += step_size / 2
    return np.arange(lower, upper, step_size) % 1.0


class hues:
    red = 0 / 360
    orange = 30 / 360
    yellow = 60 / 360
    lime = 90 / 360
    """Lime-green / yellow-green / neon-green."""
    green = 120 / 360
    teal = 150 / 360
    cyan = 180 / 360
    azure = 210 / 360
    blue = 240 / 360
    purple = 270 / 360
    magenta = 300 / 360
    pink = 330 / 360
    """Actually _hot_ pink / neon-pink, not pastel pink."""
