import numpy as np
from numpy.testing import assert_allclose

from ex_color.data.filters import levels


def test_levels_basic_mapping():
    """Test basic linear mapping."""
    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    result = levels(xs, in_low=0.0, in_high=1.0, out_low=0.0, out_high=10.0)
    assert_allclose(result, expected)


def test_levels_different_ranges():
    """Test mapping between different ranges."""
    xs = np.array([10.0, 15.0, 20.0])
    expected = np.array([0.0, 0.5, 1.0])
    result = levels(xs, in_low=10.0, in_high=20.0, out_low=0.0, out_high=1.0)
    assert_allclose(result, expected)


def test_levels_clamping_default():
    """Test default clamping behavior."""
    xs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    expected = np.array([0.0, 0.0, 5.0, 10.0, 10.0])  # Clamped at 0 and 10
    result = levels(xs, in_low=0.0, in_high=1.0, out_low=0.0, out_high=10.0)
    assert_allclose(result, expected)


def test_levels_clamping_explicit():
    """Test explicit clamping behavior."""
    xs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    expected = np.array([0.0, 0.0, 5.0, 10.0, 10.0])  # Clamped at 0 and 10
    result = levels(xs, in_low=0.0, in_high=1.0, out_low=0.0, out_high=10.0, clamp=True)
    assert_allclose(result, expected)


def test_levels_no_clamping():
    """Test behavior without clamping (extrapolation)."""
    xs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    expected = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])  # Extrapolated
    result = levels(xs, in_low=0.0, in_high=1.0, out_low=0.0, out_high=10.0, clamp=False)
    assert_allclose(result, expected)


def test_levels_degenerate_input_range():
    """Test behavior when in_low and in_high are very close."""
    xs = np.array([0.0, 0.5, 1.0])
    expected = np.array([5.0, 5.0, 5.0])  # Midpoint of out_low and out_high
    result = levels(xs, in_low=0.5, in_high=0.5 + 1e-12, out_low=0.0, out_high=10.0)
    assert_allclose(result, expected)


def test_levels_inverted_output_range():
    """Test mapping with an inverted output range."""
    xs = np.array([0.0, 0.5, 1.0])
    expected = np.array([10.0, 5.0, 0.0])
    result = levels(xs, in_low=0.0, in_high=1.0, out_low=10.0, out_high=0.0)
    assert_allclose(result, expected)


def test_levels_inverted_input_range():
    """Test mapping with an inverted input range (should still work)."""
    # This might seem weird, but it's mathematically valid.
    # Mapping x from [1, 0] to [0, 10]
    # alpha = (x - 1) / (0 - 1) = (x - 1) / -1 = 1 - x
    # result = 0 + alpha * (10 - 0) = (1 - x) * 10
    xs = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    expected = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    result = levels(xs, in_low=1.0, in_high=0.0, out_low=0.0, out_high=10.0, clamp=False)
    assert_allclose(result, expected)


def test_levels_inverted_input_range_clamped():
    """Test mapping with an inverted input range and clamping."""
    # Input range [1, 0], Output range [0, 10]
    xs = np.array([1.5, 1.0, 0.5, 0.0, -0.5])
    # alpha = (xs - 1) / (0 - 1) = 1 - xs
    # alphas before clamp: [-0.5, 0.0, 0.5, 1.0, 1.5]
    # alphas after clamp: [ 0.0, 0.0, 0.5, 1.0, 1.0]
    # result = 0 + alpha_clamped * (10 - 0)
    expected = np.array([0.0, 0.0, 5.0, 10.0, 10.0])
    result = levels(xs, in_low=1.0, in_high=0.0, out_low=0.0, out_high=10.0, clamp=True)
    assert_allclose(result, expected)
