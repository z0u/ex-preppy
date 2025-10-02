from typing import Literal

import numpy as np


def hsv_similarity(
    c1: np.ndarray,  # [..., 3]
    c2: np.ndarray,  # [..., 3]
    *,
    hemi: bool,
    mode: Literal['cosine', 'angular'],
    weight_vibrancy: bool = True,
) -> np.ndarray:
    """
    Compute a similarity between two HSV colors.

    The similarity is in [0, 1], with 1 meaning identical colors and 0 meaning maximally different colors.
    The similarity is computed as a product of:
    - hue similarity (weighted by vibrancy, since hue is undefined for grays)
    - saturation difference
    - value difference
    """
    h1, s1, v1 = np.split(c1, 3, axis=-1)
    h2, s2, v2 = np.split(c2, 3, axis=-1)

    # Compute hue similarity as the cosine of the angle between the two hues on the color wheel
    hue_diff = np.abs(h1 - h2)
    hue_angle = 360 * np.minimum(hue_diff, 1 - hue_diff)
    if mode == 'cosine':
        if hemi:
            hue_similarity = np.maximum(0, np.cos(np.radians(hue_angle)))
        else:
            hue_similarity = 0.5 + np.cos(np.radians(hue_angle)) / 2
    else:  # mode == 'angular'
        if hemi:
            hue_similarity = np.maximum(0, (90 - hue_angle) / 90)
        else:
            hue_similarity = (180 - hue_angle) / 180

    # Weight hue similarity by vibrancy (s*v) - hue only matters for vibrant colors
    r = (s1 * v1 + s2 * v2) / 2 if weight_vibrancy else 1  # average radius
    effective_hue_similarity = r * hue_similarity + (1 - r)

    similarity = effective_hue_similarity * (1 - np.abs(s1 - s2)) * (1 - np.abs(v1 - v2))
    return similarity[..., 0]
