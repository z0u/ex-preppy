import numpy as np
from pytest import approx
from ex_color.data.color_cube import ColorCube


def test_pdist_hsv():
    """Test that probability distribution function (PDF) for HSV cubes gives correct weights."""

    cube = ColorCube.from_hsv(
        h=np.linspace(0, 1, 3, endpoint=False),
        s=np.linspace(1, 0, 3),
        v=np.linspace(1, 0, 3),
    ).permute('vsh')

    # Define the weights for each cell, such that:
    # - Vibrant colors have a baseline weight of 1
    # - White: there are 3 identical "white" cells, so they should each have a weight of 1/3, to give "white" the same weight as the vibrant colors
    # - Black: there are 9 identical "black" cells, so they should each have a weight of 1/9, to give "black" the same weight as the vibrant colors
    # Other cells should be interpolated between these extremes.
    #
    # For different numbers of hues, the denominator in the top slice should be the number of hues.
    expected_weights_slice = np.array(
        [
            [
                # Top slice
                3 / 3,  # vibrant colors -> each cell = 1
                2 / 3,  # washed-out colors
                1 / 3,  # white -> full row (after broadcast) adds to 1
            ],
            [
                # Middle slice: interpolate between top and bottom
                5 / 9,  # mid-tone saturated colors
                7 / 18,  # mid-tone washed-out colors (same result interpolated top-to-bottom and side-to-side)
                2 / 9,  # middle gray
            ],
            [
                # Bottom slice -> full slice (after broadcast) adds to 1
                1 / 9,  # black
                1 / 9,  # black
                1 / 9,  # black
            ],
        ]
    )

    actual_weights = cube.bias
    expected_weights = np.broadcast_to(expected_weights_slice[:, :, np.newaxis], cube.bias.shape)

    # Normalize both actual and expected by their respective max values for comparison
    # The max expected value here is 1
    expected_weights = expected_weights / expected_weights.max(initial=1e-10)
    actual_weights = actual_weights / actual_weights.max(initial=1e-10)

    assert approx(actual_weights) == expected_weights


def test_pdist_hsv_interior():
    """Test that PDF weights are correctly interpolated for an HSV cube not touching boundaries."""

    n_hues = 3
    n_sat = 3
    n_val = 3

    # Use linspace from 0.75 to 0.25 for simpler fractions in expected weights
    s_coords = np.linspace(0.75, 0.25, n_sat)  # [0.75, 0.5, 0.25]
    v_coords = np.linspace(0.75, 0.25, n_val)  # [0.75, 0.5, 0.25]

    cube = ColorCube.from_hsv(
        h=np.linspace(0, 1, n_hues, endpoint=False),  # [0., 1/3, 2/3]
        s=s_coords,
        v=v_coords,
    ).permute('vsh')  # Shape (n_val, n_sat, n_hues) -> (3, 3, 3)

    # Expected weights derived from bilinear interpolation based on the conceptual full grid corners (see test_pdist_hsv).
    expected_weights_slice = np.array(
        [
            [47/72, 19/36, 29/72],  # V=0.75, S = [0.75, 0.5, 0.25]
            [17/36,  7/18, 11/36],  # V=0.50, S = [0.75, 0.5, 0.25]
            [ 7/24,  1/ 4,  5/24],  # V=0.25, S = [0.75, 0.5, 0.25]
        ]
    )  # fmt: skip
    # # Alternative with common denominator 72:
    # expected_weights_slice = np.array(
    #     [
    #         [47/72, 38/72, 29/72],
    #         [34/72, 28/72, 22/72],
    #         [21/72, 18/72, 15/72],
    #     ]
    # )

    # The actual bias calculation should yield these interpolated values
    actual_weights = cube.bias

    # Broadcast the 2D slice across the hue dimension
    expected_weights = np.broadcast_to(expected_weights_slice[:, :, np.newaxis], cube.bias.shape)

    # Normalize both actual and expected by their respective max values for comparison
    # The max expected value here is 47/72
    expected_weights = expected_weights / expected_weights.max(initial=1e-10)
    actual_weights = actual_weights / actual_weights.max(initial=1e-10)

    assert approx(actual_weights) == expected_weights
