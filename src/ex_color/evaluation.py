from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

from ex_color.data import ColorCube, arange_cyclic
from ex_color.data.color import get_named_colors_df
from ex_color.intervention.intervention import InterventionConfig
from ex_color.model import CNColorMLP
from ex_color.workflow import infer_with_latent_capture
import skimage as ski
from scipy.stats import pearsonr

from ex_color.data import hsv_similarity

log = logging.getLogger(__name__)


@dataclass
class TestSet:
    tags: set[str]
    color_slice_cube: ColorCube
    """Data for visualizing reconstructions across colors; typically low-res HSV"""
    loss_cube: ColorCube
    """Data for visualizing loss across colors; typically high-res HSV"""
    latent_cube: ColorCube
    """Data for visualizing latent space; typically high-res RGB"""
    named_colors: pd.DataFrame
    """Data for tabulating reconstructions on named colors; begins with columns 'name', 'rgb', 'hsv'"""

    @classmethod
    def create(
        cls,
        # Color slice cube parameters, chosen to show a representative slice of color space
        nh_color_slice=12,
        ns_color_slice=4,
        nv_color_slice=5,
        # Loss cube parameters, chosen to show smooth lines of loss
        nh_loss=150,
        ns_loss=24,
        nv_loss=24,
        # Latent cube parameters, chosen to show smooth latent shape
        n_latent=20,
        # Named colors
        n_named_hues: Literal[3, 6, 12] = 12,
        n_named_grays: Literal[3, 5] = 5,
        *,
        tags: set[str] | None = None,
    ):
        color_slice_cube = ColorCube.from_hsv(
            h=arange_cyclic(step_size=1 / nh_color_slice),
            s=np.linspace(0, 1, ns_color_slice),
            v=np.linspace(0, 1, nv_color_slice),
        )

        loss_cube = ColorCube.from_hsv(
            # Extend hue range to encompass the end pixels of the low-res cube above
            h=np.linspace(0 - 1 / nh_color_slice, 1 + 1 / nh_color_slice, nh_loss),
            s=np.linspace(0, 1, ns_loss),
            v=np.linspace(0, 1, nv_loss),
        )

        latent_cube = ColorCube.from_rgb(
            r=np.linspace(0, 1, n_latent),
            g=np.linspace(0, 1, n_latent),
            b=np.linspace(0, 1, n_latent),
        )

        named_colors = get_named_colors_df(n_hues=n_named_hues, n_grays=n_named_grays)

        return cls(
            tags=tags or set(),
            color_slice_cube=color_slice_cube,
            loss_cube=loss_cube,
            latent_cube=latent_cube,
            named_colors=named_colors,
        )

    def evaluate(self, model: CNColorMLP, interventions: Sequence[InterventionConfig], tags: set[str] | None = None):
        """Evaluate a model on all test sets, returning a new TestSet."""
        color_slice_results = evaluate_model_on_cube(model, interventions, self.color_slice_cube)
        loss_results = evaluate_model_on_cube(model, interventions, self.loss_cube)
        latent_results = evaluate_model_on_cube(model, interventions, self.latent_cube)
        named_color_results = evaluate_model_on_named_colors(model, interventions, self.named_colors)
        return type(self)(
            tags=self.tags | (tags or set()),
            color_slice_cube=color_slice_results,
            loss_cube=loss_results,
            latent_cube=latent_results,
            named_colors=named_color_results,
        )


def evaluate_model_on_cube(
    model: nn.Module,
    interventions: Sequence[InterventionConfig],
    test_data: ColorCube,
) -> ColorCube:
    """
    Evaluate model on a color cube and return reconstructions with latents and loss.

    Args:
        model: Trained model
        interventions: List of intervention configurations to apply
        test_data: ColorCube to test on

    Returns:
        ColorCube with added 'recon', 'MSE', and 'latents' variables
    """
    x = torch.tensor(test_data.rgb_grid, dtype=torch.float32)
    y, h = infer_with_latent_capture(model, x, interventions, 'bottleneck')
    per_color_loss = F.mse_loss(y, x, reduction='none').mean(dim=-1)
    return test_data.assign(
        recon=y.numpy().reshape((*test_data.shape, -1)),
        MSE=per_color_loss.numpy().reshape((*test_data.shape, -1)),
        latents=h.numpy().reshape((*test_data.shape, -1)),
    )


def evaluate_model_on_named_colors(
    model: nn.Module,
    interventions: Sequence[InterventionConfig],
    test_data: 'pd.DataFrame',
):
    """
    Evaluate model on named colors and return reconstructions with loss.

    Args:
        model: Trained model
        interventions: List of intervention configurations to apply
        test_data: DataFrame with 'rgb' column containing RGB tuples

    Returns:
        DataFrame with added 'recon' and 'MSE' columns
    """
    x = torch.tensor(test_data['rgb'], dtype=torch.float32)
    y, _ = infer_with_latent_capture(model, x, interventions, 'bottleneck')
    per_color_loss = F.mse_loss(y, x, reduction='none').mean(dim=-1)
    y_tuples = [row.tolist() for row in y.numpy()]
    return test_data.assign(recon=y_tuples, MSE=per_color_loss.numpy())


def hstack_named_results(baseline: TestSet, *res: TestSet) -> pd.DataFrame:
    """
    Create a table of results across several experiments.

    Returns a DataFrame with columns: ['name', 'rgb', 'hsv', 'baseline',
    '{tag}', '{tag}-delta', ...], where 'baseline' is the 'MSE' from the
    baseline TestSet, '{tag}' is the 'MSE' from each subsequent TestSet, and
    '{tag}-delta' is the difference between that 'MSE' and the baseline.
    """
    names = [' '.join(r.tags) for r in res]
    df = baseline.named_colors[['name', 'rgb', 'hsv', 'MSE']].rename(columns={'MSE': 'baseline'})
    for name, r in zip(names, res, strict=True):
        df = df.merge(
            r.named_colors[['name', 'MSE']].rename(columns={'MSE': name}),
            on='name',
        )
        df[f'{name}-delta'] = df[name] - df['baseline']
    return df


def error_correlation(
    data: ColorCube | TestSet, anchor_hsv: tuple[float, float, float], *, power: float
) -> tuple[float, float]:
    """Compute correlation between similarity to anchor and reconstruction error."""
    # Use loss_cube because its coordinates are RGB (by convention) so it is unbiased
    cube = data.loss_cube if isinstance(data, TestSet) else data
    cube = cube.assign(hsv=ski.color.rgb2hsv(cube['color']))
    cube = cube.assign(similarity=hsv_similarity(cube['hsv'], np.array(anchor_hsv), hemi=True, mode='angular') ** power)
    corr, p_value = pearsonr(cube['similarity'].flatten(), cube['MSE'].flatten())
    return float(corr), float(p_value)
