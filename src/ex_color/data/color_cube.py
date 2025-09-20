from __future__ import annotations

from typing import SupportsIndex, overload

import numpy as np

from ex_color.data.grid import coordinate_grid

color_axes = {
    'r': 'red',
    'g': 'green',
    'b': 'blue',
    'h': 'hue',
    's': 'saturation',
    'v': 'value',
}


class ColorCube:
    """A tensor of RGB values with coordinates in various color spaces (e.g. HSV)."""

    coordinates: tuple[np.ndarray, ...]
    """The coordinate axes of the cube, with shapes [a,], [b,], [c,]."""
    space: str
    """The color space of the cube [e.g. 'vsh'], which is some permutation of the canonical space."""
    canonical_space: str
    """The well-known color space for the cube [e.g. 'hsv'], regardless of the current axis order."""

    def __init__(
        self,
        vars: dict[str, np.ndarray],
        coordinates: tuple[np.ndarray, ...],
        space: str,
        canonical_space: str,
    ):
        if sorted(space) != sorted(canonical_space):
            raise ValueError(f'Cannot create ColorCube with different spaces: {space} != {canonical_space}')
        coord_shape = tuple(len(coord) for coord in coordinates)

        if 'color' not in vars:
            raise ValueError('Variables must include a "color" array')
        if len(vars['color'].shape) != len(space) + 1:
            raise ValueError(f'"color" variable must have {len(space) + 1} dimensions (vector)')
        if vars['color'].shape[-1] != 3:
            raise ValueError('"color" grid must have three channels')

        if 'bias' not in vars:
            raise ValueError('Variables must include a "bias" array')

        for k, v in vars.items():
            grid_shape = v.shape[: len(coord_shape)]
            if coord_shape != grid_shape:
                raise ValueError(f'Variable "{k}" must match coordinates: {grid_shape} != {coord_shape}')

        self.space = space
        self.shape = coord_shape
        self.canonical_space = canonical_space
        self.coordinates = coordinates
        self.vars = vars

    @overload
    def __getitem__(self, key: str, /) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: tuple[np._ArrayInt_co, ...], /) -> ColorCube: ...
    @overload
    def __getitem__(self, key: tuple[SupportsIndex, ...], /) -> ColorCube: ...
    @overload
    def __getitem__(self, key: tuple[np._ToIndex, ...], /) -> ColorCube: ...
    def __getitem__(
        self,
        key: str | tuple[np._ArrayInt_co, ...] | tuple[SupportsIndex, ...] | tuple[np._ToIndex, ...],
        /,
    ) -> np.ndarray | ColorCube:
        """
        Get data from the cube.

        Args:
            key: Variable name or coordinate slices.

        Return: The named variable (array) or a sliced cube as a view of the current one.
        """
        if isinstance(key, str):
            return self.vars[key]

        if len(key) != len(self.space):
            raise ValueError(f'Expected {len(self.space)} slices, got {len(key)}')
        new_vars = {k: v.__getitem__(key) for k, v in self.vars.items()}
        new_coords = tuple(
            c[k]  #
            for c, k in zip(self.coordinates, key, strict=True)
        )
        return type(self)(new_vars, new_coords, self.space, self.canonical_space)

    @overload
    def assign(self, name: str, var: np.ndarray, /) -> ColorCube: ...
    @overload
    def assign(self, /, **kwargs: np.ndarray) -> ColorCube: ...
    def assign(self, name: str | None = None, var: np.ndarray | None = None, /, **kwargs) -> ColorCube:
        """
        Assign a new variable to the cube.

        Args:
            name: The name of the new variable.
            var: The variable array to assign. Must match the cube's grid shape.
            **kwargs: Named variables to assign.

        Returns a new cube with all original variables in addition to new ones. Existing variables can be re-assigned.
        """
        if name is not None:
            assert var is not None
            return type(self)({**self.vars, name: var}, self.coordinates, self.space, self.canonical_space)
        else:
            return type(self)({**self.vars, **kwargs}, self.coordinates, self.space, self.canonical_space)

    @property
    def rgb_grid(self) -> np.ndarray:
        """The RGB values in the cube, with shape [a, b, c, 3]."""
        return self.vars['color']

    @property
    def bias(self) -> np.ndarray:
        """The probability distribution weights for the cube, with shape [a, b, c]."""
        return self.vars['bias']

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

        return cls({'color': grid, 'bias': bias}, (h, s, v), 'hsv', 'hsv')

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
        return cls({'color': grid, 'bias': bias}, (r, g, b), 'rgb', 'rgb')

    def permute(self, new_space: str):
        """Re-order the axes of the ColorCube."""
        indices = self.transpose_idx(new_space)
        new_vars = {
            k: np.transpose(a, indices) if len(a.shape) == len(indices) else np.transpose(a, indices + (-1,))  #
            for k, a in self.vars.items()
        }
        new_coordinates = tuple(self.coordinates[i] for i in indices)
        return ColorCube(new_vars, new_coordinates, new_space, self.canonical_space)

    def transpose_idx(self, new_space: str) -> tuple[int, int, int]:
        """Get the dimension indices that would transpose this cube to a target space."""
        if set(self.space) != set(new_space):
            raise ValueError(f'Cannot permute {self.space} to {new_space}: different axes')
        indices = tuple(self.space.index(axis) for axis in new_space)
        assert len(indices) == 3
        return indices
