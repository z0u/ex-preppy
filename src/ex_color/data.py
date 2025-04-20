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
    bias: np.ndarray
    """The probability distribution weights for the cube, with shape (a, b, c)."""

    def __init__(
        self,
        rgb_grid: np.ndarray,
        bias: np.ndarray,
        coordinates: tuple[np.ndarray, ...],
        space: str,
        canonical_space: str,
    ):
        if sorted(space) != sorted(canonical_space):
            raise ValueError(f'Cannot create ColorCube with different spaces: {space} != {canonical_space}')
        if rgb_grid.shape[-1] != 3:
            raise ValueError(f'RGB grid must have three channels: {rgb_grid.shape[-1]} != 3')
        grid_shape = rgb_grid.shape[:-1]
        if grid_shape != bias.shape:
            raise ValueError(f'RGB grid and bias must have the same shape: {grid_shape} != {bias.shape}')
        coord_shape = tuple(len(coord) for coord in coordinates)
        if grid_shape != coord_shape:
            raise ValueError(f'RGB grid shape {grid_shape} does not match coordinates {coord_shape}')

        self.space = space
        self.canonical_space = canonical_space
        self.coordinates = coordinates
        self.rgb_grid = rgb_grid
        self.bias = bias

    @classmethod
    def from_hsv(cls, h: np.ndarray, s: np.ndarray, v: np.ndarray):
        """
        Create a ColorCube from HSV values.

        Args:
            h: Hue values (0-1).
            s: Saturation values (0-1).
            v: Value (brightness) values (0-1).

        Returns:
            A ColorCube object.

        Notes:
            - The hue values are cyclic, so they wrap around at 1.0.
            - The saturation and value values are not cyclic, so they are clamped between 0 and 1.
            - The bias is calculated based on the saturation and value coordinates using bilinear interpolation, such that:
                - Vibrant colors (S=1, V=1) have a bias of 1.
                - White (S=0, V=1) has a bias of 1 / n_hues.
                - Black (S=any, V=0) has a bias of 1 / (n_sat * n_hues).
        """
        import skimage as ski

        hsv_grid_coords = coordinate_grid(h, s, v)
        grid = ski.color.hsv2rgb(hsv_grid_coords)

        # Calculate bias based on S and V coordinates using bilinear interpolation
        # Corner weights:
        # w(S=1, V=1) = 1 (Vibrant)
        # w(S=0, V=1) = 1 / n_hues (White)
        # w(S=any, V=0) = 1 / (n_sat * n_hues) (Black)
        n_hues = max(1, len(h))  # Avoid division by zero if len is 0
        n_sat = max(1, len(s))  # Avoid division by zero if len is 0

        # Extract S and V coordinates from the grid
        # hsv_grid_coords has shape (n_hues, n_sat, n_val, 3)
        # We need S and V grids with shape (n_hues, n_sat, n_val)
        S_grid = hsv_grid_coords[:, :, :, 1]
        V_grid = hsv_grid_coords[:, :, :, 2]

        # Apply the bilinear interpolation formula
        w_black = 1.0 / (n_sat * n_hues)
        w_white = 1.0 / n_hues
        w_vibrant = 1.0

        # w(s, v) = w_black * (1-v) + w_white * (1-s) * v + w_vibrant * s * v
        bias = w_black * (1 - V_grid) + w_white * (1 - S_grid) * V_grid + w_vibrant * S_grid * V_grid

        return cls(grid, bias, (h, s, v), 'hsv', 'hsv')

    @classmethod
    def from_rgb(cls, r: np.ndarray, g: np.ndarray, b: np.ndarray):
        """
        Create a ColorCube from RGB values.

        Args:
            r: Red values (0-1).
            g: Green values (0-1).
            b: Blue values (0-1).

        Returns:
            A ColorCube object.

        Notes:
            - The RGB values are not cyclic, so they are clamped between 0 and 1.
            - The bias is assumed to be uniform across the RGB space, so it is set to 1 for all coordinates.
        """
        grid = coordinate_grid(r, g, b)
        bias = np.ones(grid.shape[:-1], dtype=float)
        return cls(grid, bias, (r, g, b), 'rgb', 'rgb')

    def permute(self, new_space: str):
        """Re-order the axes of the ColorCube."""
        if set(self.space) != set(new_space):
            raise ValueError(f'Cannot permute {self.space} to {new_space}: different axes')
        indices = tuple(self.space.index(axis) for axis in new_space)
        new_grid = np.transpose(self.rgb_grid, indices + (-1,))
        new_bias = np.transpose(self.bias, indices)
        new_coordinates = tuple(self.coordinates[i] for i in indices)
        return ColorCube(new_grid, new_bias, new_coordinates, new_space, self.canonical_space)


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
