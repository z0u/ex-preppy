import numpy as np
import pytest

from ex_color.data.color_cube import ColorCube
from ex_color.data.cube_sampler import CubeRandomSampler, vibrancy


@pytest.fixture
def hsv_cube_2x2x2() -> ColorCube:
    """Creates a simple 2x2x2 HSV ColorCube for testing."""
    # Use from_hsv which calculates bias correctly
    h = np.linspace(0, 1, 2, endpoint=False)  # Hue is cyclic
    s = np.linspace(0, 1, 2)
    v = np.linspace(0, 1, 2)
    cube = ColorCube.from_hsv(h, s, v)

    # Overwrite the default bias with a non-uniform one for testing focus
    bias = np.arange(8).reshape(2, 2, 2).astype(float)
    bias /= bias.sum()  # Normalize
    cube.bias = bias
    return cube


@pytest.fixture
def rgb_cube_2x2x2() -> ColorCube:
    """Creates a simple 2x2x2 RGB ColorCube for testing error conditions."""
    # Use from_rgb which sets uniform bias
    coords = [np.linspace(0, 1, 2)] * 3
    return ColorCube.from_rgb(*coords)


def test_sampler_initialization(hsv_cube_2x2x2: ColorCube):
    """Test basic initialization of the CubeRandomSampler."""
    cube = hsv_cube_2x2x2
    sampler = CubeRandomSampler(cube)

    assert sampler.cube is cube
    assert sampler.original_shape == (2, 2, 2)
    assert sampler.n_points == 8
    assert isinstance(sampler.rng, np.random.Generator)
    assert sampler.flat_rgb.shape == (8, 3)
    # Check initial effective bias (should be proportional to cube.bias)
    assert sampler._effective_bias is not None
    np.testing.assert_allclose(sampler._effective_bias, cube.bias.flatten())
    np.testing.assert_allclose(np.sum(sampler._effective_bias), 1.0)


def test_set_focus_weights_normal(hsv_cube_2x2x2: ColorCube):
    """Test setting focus weights and recalculating effective bias."""
    cube = hsv_cube_2x2x2
    sampler = CubeRandomSampler(cube)

    # Uniform focus weights should result in bias proportional to cube.bias
    weights_uniform = np.ones_like(cube.bias)
    sampler.set_focus_weights(weights_uniform)
    assert sampler._effective_bias is not None
    np.testing.assert_allclose(sampler._effective_bias, cube.bias.flatten())

    # Focus on a single point (e.g., index 3)
    weights_single = np.zeros_like(cube.bias)
    weights_single.flat[3] = 1.0
    sampler.set_focus_weights(weights_single)
    assert sampler._effective_bias is not None
    expected_bias = np.zeros_like(cube.bias).flatten()
    # Effective bias = weights * cube.bias, then normalized
    # Only point 3 has weight, so only its original bias matters
    expected_bias[3] = cube.bias.flat[3]
    expected_bias /= expected_bias.sum()  # Normalize
    np.testing.assert_allclose(sampler._effective_bias, expected_bias)


def test_set_focus_weights_zero_bias(hsv_cube_2x2x2: ColorCube):
    """Test setting focus weights that result in zero effective bias."""
    cube = hsv_cube_2x2x2
    sampler = CubeRandomSampler(cube)

    # Set weights that are zero where bias is non-zero
    weights_zero = np.zeros_like(cube.bias)
    # If cube.bias itself has zeros, this might still work, but our fixture bias is all non-zero
    sampler.set_focus_weights(weights_zero)
    assert sampler._effective_bias is None

    # Test with negative weights (should be clipped by np.maximum)
    weights_negative = -np.ones_like(cube.bias)
    sampler.set_focus_weights(weights_negative)
    assert sampler._effective_bias is None


def test_sample_no_bias(hsv_cube_2x2x2: ColorCube):
    """Test that sampling raises ValueError when effective bias is None."""
    cube = hsv_cube_2x2x2
    sampler = CubeRandomSampler(cube)

    # Force zero effective bias
    sampler.set_focus_weights(np.zeros_like(cube.bias))
    assert sampler._effective_bias is None

    with pytest.raises(ValueError, match='No points available for sampling.'):
        sampler.sample(10)


def test_sample_basic(hsv_cube_2x2x2: ColorCube):
    """Test basic sampling returns correct shape and count."""
    cube = hsv_cube_2x2x2
    sampler = CubeRandomSampler(cube)
    k = 50

    samples = sampler.sample(k)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k, 3)
    # Check if sampled colors are actually from the cube's RGB grid
    # This is a bit tricky due to potential floating point comparisons
    # Let's check if each sampled row exists in the flat_rgb array
    flat_rgb_set = {tuple(row) for row in sampler.flat_rgb}
    sampled_tuples = {tuple(row) for row in samples}
    assert sampled_tuples.issubset(flat_rgb_set)


def test_sample_single_point_bias(hsv_cube_2x2x2: ColorCube):
    """Test sampling when bias is concentrated on a single point."""
    cube = hsv_cube_2x2x2
    sampler = CubeRandomSampler(cube)

    # Focus bias entirely on the point with index 3
    weights_single = np.zeros_like(cube.bias)
    weights_single.flat[3] = 1.0
    sampler.set_focus_weights(weights_single)

    assert sampler._effective_bias is not None
    expected_bias = np.zeros(sampler.n_points)
    expected_bias[3] = 1.0
    np.testing.assert_allclose(sampler._effective_bias, expected_bias)

    k = 20
    samples = sampler.sample(k)
    expected_rgb = sampler.flat_rgb[3]

    assert samples.shape == (k, 3)
    # All samples should be the RGB value of the single focused point
    np.testing.assert_allclose(samples, np.tile(expected_rgb, (k, 1)))


# --- Tests for vibrancy function ---


def test_vibrancy_hsv(hsv_cube_2x2x2: ColorCube):
    """Test the vibrancy function with a valid HSV cube and explicit values."""
    cube = hsv_cube_2x2x2
    vib = vibrancy(cube)

    assert vib.shape == cube.shape  # Use the new shape property
    assert np.all(vib >= 0) and np.all(vib <= 1)

    # Expected vibrancy = S * V for the 2x2x2 grid (H, S, V axes)
    # H=[0.0, 0.5], S=[0.0, 1.0], V=[0.0, 1.0]
    # Slice H=0.0:
    #   S=0.0, V=0.0 -> 0.0 * 0.0 = 0.0
    #   S=0.0, V=1.0 -> 0.0 * 1.0 = 0.0
    #   S=1.0, V=0.0 -> 1.0 * 0.0 = 0.0
    #   S=1.0, V=1.0 -> 1.0 * 1.0 = 1.0
    # Slice H=0.5:
    #   S=0.0, V=0.0 -> 0.0 * 0.0 = 0.0
    #   S=0.0, V=1.0 -> 0.0 * 1.0 = 0.0
    #   S=1.0, V=0.0 -> 1.0 * 0.0 = 0.0
    #   S=1.0, V=1.0 -> 1.0 * 1.0 = 1.0
    expected_vibrancy = np.array(
        [
            [
                [0.0, 0.0],  # H=0.0, S=0.0, V=[0,1]
                [0.0, 1.0],
            ],  # H=0.0, S=1.0, V=[0,1]
            [
                [0.0, 0.0],  # H=0.5, S=0.0, V=[0,1]
                [0.0, 1.0],
            ],  # H=0.5, S=1.0, V=[0,1]
        ]
    )

    np.testing.assert_allclose(vib, expected_vibrancy)


def test_vibrancy_non_hsv_error(rgb_cube_2x2x2: ColorCube):
    """Test vibrancy raises ValueError for non-HSV cubes."""
    with pytest.raises(ValueError, match='Cannot create vibrant focus for non-HSV cube'):
        vibrancy(rgb_cube_2x2x2)
