import numpy as np


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


def arange_cyclic(
    lower: float = 0.0,
    upper: float | None = None,
    step_size: float = 1 / 12,
    period: float = 1.0,
    inclusive: bool = False,
) -> np.ndarray:
    """Generate an array of values over a periodic range."""
    lower = lower % period
    if upper is None:
        upper = lower + period
    else:
        upper = upper % period
    if lower > upper:
        upper += period
    if inclusive:
        upper += step_size / 2
    return np.arange(lower, upper, step_size) % period
