import numpy as np


def levels(
    xs: np.ndarray,
    in_low=0.0,
    in_high=1.0,
    out_low=0.0,
    out_high=1.0,
    clamp=True,
) -> np.ndarray:
    """
    Map values from one range to another.

    Behaves like the Levels filter in Krita and Photoshop.
    """
    if abs(in_low - in_high) < 1e-10:
        # Degenerate to constant value
        return np.full_like(xs, (out_low + out_high) / 2)

    alpha = (xs - in_low) / (in_high - in_low)
    if clamp:
        alpha = np.clip(alpha, 0, 1)
    return out_low + alpha * (out_high - out_low)
