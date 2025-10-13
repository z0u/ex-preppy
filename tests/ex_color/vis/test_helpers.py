"""Tests for the vis helpers module."""

import numpy as np

from ex_color.data.color_cube import ColorCube
from ex_color.evaluation import Resultset
from ex_color.vis.helpers import hstack_named_results, tags_for_file


def test_tags_for_file():
    """Test tags_for_file converts tags to filename-safe strings."""
    assert tags_for_file(['Simple']) == 'simple'
    assert tags_for_file(['No Intervention']) == 'no-intervention'
    assert tags_for_file(['test', 'case']) == 'test-case'
    assert tags_for_file(['Multiple  Spaces']) == 'multiple-spaces'
    assert tags_for_file(['Special!@#Characters']) == 'special-characters'


def test_hstack_named_results():
    """Test hstack_named_results creates comparison table."""
    import pandas as pd

    # Create mock data
    colors_df1 = pd.DataFrame(
        {
            'name': ['red', 'green', 'blue'],
            'rgb': [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            'hsv': [(0, 1, 1), (0.33, 1, 1), (0.66, 1, 1)],
            'MSE': [0.1, 0.2, 0.3],
        }
    )

    colors_df2 = pd.DataFrame(
        {
            'name': ['red', 'green', 'blue'],
            'MSE': [0.15, 0.25, 0.35],
        }
    )

    # Create mock resultsets
    cube = ColorCube.from_rgb(r=np.array([0, 1]), g=np.array([0, 1]), b=np.array([0, 1]))
    resultset1 = Resultset(
        tags=['baseline'],
        latent_cube=cube,
        color_slice_cube=cube,
        loss_cube=cube,
        named_colors=colors_df1,
    )
    resultset2 = Resultset(
        tags=['intervention'],
        latent_cube=cube,
        color_slice_cube=cube,
        loss_cube=cube,
        named_colors=colors_df2,
    )

    result = hstack_named_results(resultset1, resultset2)

    assert 'name' in result.columns
    assert 'baseline' in result.columns
    assert 'intervention' in result.columns
    assert 'intervention-delta' in result.columns
    assert len(result) == 3

    # Check delta calculation
    assert (result['intervention-delta'] == result['intervention'] - result['baseline']).all()
