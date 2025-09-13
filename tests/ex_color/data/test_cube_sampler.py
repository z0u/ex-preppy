import numpy as np
import pytest

from ex_color.data.color_cube import ColorCube
from ex_color.data.cube_sampler import CubeRandomSampler, vibrancy


@pytest.fixture
def hsv_cube_3x3x3() -> ColorCube:
    """Creates a simple 3x3x3 HSV ColorCube for testing."""
    # Use from_hsv which calculates bias correctly
    h = np.linspace(0, 1, 3, endpoint=False)  # Hue is cyclic
    s = np.linspace(0, 1, 3)
    v = np.linspace(0, 1, 3)
    return ColorCube.from_hsv(h, s, v)


@pytest.fixture
def rgb_cube_3x3x3() -> ColorCube:
    """Creates a simple 3x3x3 RGB ColorCube for testing error conditions."""
    # Use from_rgb which sets uniform bias
    coords = [np.linspace(0, 1, 3)] * 3
    return ColorCube.from_rgb(*coords)


def test_sampler_initialization(hsv_cube_3x3x3: ColorCube):
    """Test basic initialization of the CubeRandomSampler."""
    cube = hsv_cube_3x3x3
    sampler = CubeRandomSampler(cube)

    assert sampler.cube is cube
    assert sampler.original_shape == (3, 3, 3)
    assert sampler.n_points == 27
    assert isinstance(sampler.rng, np.random.Generator)
    assert sampler.flat_rgb.shape == (27, 3)
    # Check initial effective bias (should be proportional to cube.bias)
    assert sampler._effective_bias is not None
    norm_bias = cube.bias.flatten() / np.sum(cube.bias)
    np.testing.assert_allclose(sampler._effective_bias, norm_bias)
    np.testing.assert_allclose(np.sum(sampler._effective_bias), 1.0)


def test_set_focus_weights_zero_bias(hsv_cube_3x3x3: ColorCube):
    """Test setting focus weights that result in zero effective bias."""
    cube = hsv_cube_3x3x3
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


def test_sample_no_bias(hsv_cube_3x3x3: ColorCube):
    """Test that sampling raises ValueError when effective bias is None."""
    cube = hsv_cube_3x3x3
    sampler = CubeRandomSampler(cube)

    # Force zero effective bias
    sampler.set_focus_weights(np.zeros_like(cube.bias))
    assert sampler._effective_bias is None

    with pytest.raises(ValueError, match='No points available for sampling.'):
        sampler.sample(10)


def test_sample_basic(hsv_cube_3x3x3: ColorCube):
    """Test basic sampling returns correct shape and count."""
    cube = hsv_cube_3x3x3
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


def test_sample_single_point_bias(hsv_cube_3x3x3: ColorCube):
    """Test sampling when bias is concentrated on a single point."""
    cube = hsv_cube_3x3x3
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


def test_vibrancy_hsv(hsv_cube_3x3x3: ColorCube):
    """Test the vibrancy function with a valid HSV cube and explicit values."""
    cube = hsv_cube_3x3x3
    vib = vibrancy(cube)

    assert vib.shape == cube.shape
    assert np.all(vib >= 0) and np.all(vib <= 1)

    # Expected vibrancy = S * V for the 3x3x3 grid (H, S, V axes)
    expected_vibrancy = np.array(
        [
            [
                [0 / 4, 0 / 4, 0 / 4],  # H=..., S=0.0, V=[0, 0.5, 1]
                [0 / 4, 1 / 4, 2 / 4],  # H=..., S=0.5, V=[0, 0.5, 1]
                [0 / 4, 2 / 4, 4 / 4],  # H=..., S=1.0, V=[0, 0.5, 1]
            ],
        ]
        * 3  # H=[0.0, 0.333, 0.666] (vibrancy is independent of H)
    )

    np.testing.assert_allclose(vib, expected_vibrancy)


def test_vibrancy_non_hsv_error(rgb_cube_3x3x3: ColorCube):
    """Test vibrancy raises ValueError for non-HSV cubes."""
    cube = rgb_cube_3x3x3
    vib = vibrancy(cube)

    assert vib.shape == cube.shape
    assert np.all(vib >= 0) and np.all(vib <= 1)

    # Expected vibrancy:
    # - 0 at black, white, gray
    # - 1 at primary and secondary colors (other corners)
    # - 1 at ternaries (edges between primaries and secondaries)
    # - 0.5 half-way between the ones and zeroes.
    expected_vibrancy = np.array(
        [
            [
                [0.0, 0.5, 1.0],  # zero at black
                [0.5, 0.5, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [0.5, 0.5, 1.0],
                [0.5, 0.0, 0.5],  # zero at gray
                [1.0, 0.5, 0.5],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.5, 0.5],
                [1.0, 0.5, 0.0],  # zero at white
            ],
        ]
    )

    np.testing.assert_allclose(vib, expected_vibrancy)
