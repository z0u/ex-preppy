import numpy as np

from ex_color.data.color import hues
from ex_color.data.cyclic import isbetween_cyclic


def test_isbetween_cyclic_regular_case():
    """Test isbetween_cyclic with regular ranges where lower < upper."""
    # Value inside range
    assert isbetween_cyclic(0.5, 0.3, 0.7)
    assert np.all(isbetween_cyclic(np.array([0.4, 0.5, 0.6]), 0.3, 0.7))

    # Value outside range
    assert not isbetween_cyclic(0.2, 0.3, 0.7)
    assert not isbetween_cyclic(0.8, 0.3, 0.7)
    assert not np.any(isbetween_cyclic(np.array([0.1, 0.2, 0.8, 0.9]), 0.3, 0.7))


def test_isbetween_cyclic_wrap_around():
    """Test isbetween_cyclic with wrap-around ranges where lower > upper."""
    # Values in wrap-around range
    assert isbetween_cyclic(0.9, 0.7, 0.2)
    assert isbetween_cyclic(0.1, 0.7, 0.2)
    assert np.all(isbetween_cyclic(np.array([0.8, 0.9, 0.0, 0.1]), 0.7, 0.2))

    # Values outside wrap-around range
    assert not isbetween_cyclic(0.5, 0.7, 0.2)
    assert not np.any(isbetween_cyclic(np.array([0.3, 0.4, 0.5, 0.6]), 0.7, 0.2))


def test_isbetween_cyclic_edge_cases():
    """Test isbetween_cyclic with edge cases."""
    # Exactly at boundaries
    assert isbetween_cyclic(0.3, 0.3, 0.7)
    assert isbetween_cyclic(0.7, 0.3, 0.7)

    # Wrap-around edge cases
    assert isbetween_cyclic(0.7, 0.7, 0.2)
    assert isbetween_cyclic(0.2, 0.7, 0.2)

    # Equal bounds
    assert isbetween_cyclic(0.5, 0.5, 0.5)  # Should be true when all are equal


def test_isbetween_cyclic_with_hues():
    """Test isbetween_cyclic using the hues enum values."""
    # Check if colors are in expected ranges
    assert isbetween_cyclic(hues.yellow, hues.orange, hues.green)
    assert isbetween_cyclic(hues.blue, hues.cyan, hues.purple)

    # Wrap-around test with hues
    assert isbetween_cyclic(hues.red, hues.purple, hues.orange)
    assert isbetween_cyclic(hues.magenta, hues.purple, hues.orange)

    # Color not in range
    assert not isbetween_cyclic(hues.green, hues.purple, hues.orange)


def test_isbetween_cyclic_different_period():
    """Test isbetween_cyclic with different period values."""
    # Testing with period=2.0
    assert isbetween_cyclic(1.5, 1.0, 1.8, period=2.0)
    assert not isbetween_cyclic(0.5, 1.0, 1.8, period=2.0)

    # Wrap-around with period=2.0
    assert isbetween_cyclic(0.2, 1.5, 0.5, period=2.0)
    assert isbetween_cyclic(1.9, 1.5, 0.5, period=2.0)
    assert not isbetween_cyclic(1.0, 1.5, 0.5, period=2.0)

    # Testing with period=360 (for degrees)
    assert isbetween_cyclic(45, 30, 60, period=360)
    assert isbetween_cyclic(350, 300, 30, period=360)  # Wrap around
