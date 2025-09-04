import numpy as np
import numpy.testing as npt
import pytest

from ex_color.data.color_cube import ColorCube


def make_cube(a: int = 3, b: int = 4, c: int = 2) -> ColorCube:
    # Coordinates
    h = np.linspace(0.0, 1.0, a, endpoint=False)
    s = np.linspace(1.0, 0.0, b)
    v = np.linspace(0.2, 0.8, c)

    # Construct easy-to-verify grids
    i_idx = np.arange(a)[:, None, None]
    J = np.arange(b)[None, :, None]
    K = np.arange(c)[None, None, :]

    # color[a,b,c,3]: encode axis indices into channels
    ch0 = np.broadcast_to(i_idx, (a, b, c))  # depends on i
    ch1 = np.broadcast_to(J, (a, b, c))  # depends on j
    ch2 = np.broadcast_to(K, (a, b, c))  # depends on k
    color = np.stack([ch0, ch1, ch2], axis=-1).astype(float)

    # bias[a,b,c]: unique per-cell id for verification
    bias = (i_idx + 10 * J + 100 * K).astype(float).reshape(a, b, c)

    vars = {
        'color': color,
        'bias': bias,
    }
    return ColorCube(vars, (h, s, v), 'hsv', 'hsv')


def test_getitem_variable_name_returns_array():
    cube = make_cube()
    arr = cube['color']
    # Same object returned
    assert arr is cube.vars['color']
    assert arr.shape == (3, 4, 2, 3)


def test_getitem_slice_returns_cube_and_views():
    cube = make_cube(a=4, b=5, c=3)
    key = (slice(1, 3), slice(None), slice(0, 1))  # keep dims
    sub = cube[key]
    assert isinstance(sub, ColorCube)

    # Space metadata preserved
    assert sub.space == cube.space
    assert sub.canonical_space == cube.canonical_space

    # Coordinates sliced correctly and remain arrays
    assert tuple(len(ax) for ax in sub.coordinates) == (2, 5, 1)
    npt.assert_allclose(sub.coordinates[0], cube.coordinates[0][1:3])
    npt.assert_allclose(sub.coordinates[1], cube.coordinates[1][:])
    npt.assert_allclose(sub.coordinates[2], cube.coordinates[2][0:1])

    # Vars sliced with matching shapes; bias/color are views where possible
    assert sub.vars['color'].shape == (2, 5, 1, 3)
    assert sub.vars['bias'].shape == (2, 5, 1)
    assert np.shares_memory(sub.vars['bias'], cube.vars['bias'])
    assert np.shares_memory(sub.vars['color'], cube.vars['color'])

    # Content matches original region
    npt.assert_allclose(sub.vars['bias'], cube.vars['bias'][1:3, :, 0:1])
    npt.assert_allclose(sub.vars['color'], cube.vars['color'][1:3, :, 0:1, :])


def test_getitem_wrong_number_of_slices_raises():
    cube = make_cube()
    with pytest.raises(ValueError):
        _ = cube[(slice(None), slice(None))]  # only 2 slices for 3D cube
