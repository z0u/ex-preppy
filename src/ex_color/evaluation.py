from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence
from warnings import deprecated

import numpy as np
import pandas as pd
import skimage as ski
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.nn import functional as F

from ex_color.data import ColorCube, arange_cyclic, hsv_similarity
from ex_color.data.color import get_named_colors_df
from ex_color.intervention.intervention import InterventionConfig
from ex_color.model import CNColorMLP
from ex_color.workflow import infer_with_latent_capture

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


def hstack_named_results(baseline: TestSet | pd.DataFrame, *res: TestSet) -> pd.DataFrame:
    """
    Create a table of results across several experiments.

    Returns a DataFrame with columns: ['name', 'rgb', 'hsv', 'baseline',
    '{tag}', '{tag}-delta', ...], where 'baseline' is the 'MSE' from the
    baseline TestSet, '{tag}' is the 'MSE' from each subsequent TestSet, and
    '{tag}-delta' is the difference between that 'MSE' and the baseline.
    """
    baseline = baseline.named_colors if isinstance(baseline, TestSet) else baseline
    names = [' '.join(r.tags) for r in res]
    df = baseline[['name', 'rgb', 'hsv', 'MSE']].rename(columns={'MSE': 'baseline'})
    if 'similarity-metric' in baseline.columns:
        df['similarity-metric'] = baseline['similarity-metric']
    for name, r in zip(names, res, strict=True):
        df = df.merge(
            r.named_colors[['name', 'MSE']].rename(columns={'MSE': name}),
            on='name',
        )
        df[f'{name}-delta'] = df[name] - df['baseline']
    return df


def error_correlation(
    data: ColorCube | TestSet,
    anchor_hsv: tuple[float, float, float],
    *,
    power: float,
) -> tuple[float, float]:
    """Compute correlation between similarity to anchor and reconstruction error."""
    # Use loss_cube because its coordinates are RGB (by convention) so it is unbiased
    cube = data.loss_cube if isinstance(data, TestSet) else data
    cube = cube.assign(hsv=ski.color.rgb2hsv(cube['color']))
    cube = cube.assign(similarity=hsv_similarity(cube['hsv'], np.array(anchor_hsv), hemi=True, mode='angular') ** power)
    corr, p_value = pearsonr(cube['similarity'].flatten(), cube['MSE'].flatten())
    return float(corr), float(p_value)


class EvaluationPlan:
    def __init__(
        self,
        tags: set[str],
        transform: Callable[[CNColorMLP], CNColorMLP],
        interventions: Sequence[InterventionConfig],
    ):
        self.tags = tags
        self.interventions = interventions
        self.transform = transform

    def evaluate(self, model: CNColorMLP, data: ColorCube) -> ColorCube:
        return evaluate_model_on_cube(model, self.interventions, data)


class ScoreByHSVSimilarity:
    def __init__(
        self,
        evaluation_plan: EvaluationPlan,
        anchor_hsv: tuple[float, float, float],
        power: float,
        cube_subdivisions: int,
    ):
        self.evaluation_plan = evaluation_plan
        self.anchor_hsv = anchor_hsv
        self.power = power
        coords = np.linspace(0, 1, cube_subdivisions)
        self.val_data = ColorCube.from_rgb(r=coords, g=coords, b=coords)

    def __call__(self, model: CNColorMLP) -> float:
        transformed_model = self.evaluation_plan.transform(model)
        results = evaluate_model_on_cube(transformed_model, self.evaluation_plan.interventions, self.val_data)
        r, p_value = error_correlation(results, self.anchor_hsv, power=self.power)
        del p_value
        return r**2


@deprecated('Use Result instead')
@dataclass
class BaseResult:
    score: float

    def __gt__(self, other: BaseResult | None) -> bool:
        if other is None:
            return True
        return self.score > other.score


@dataclass
class Result:
    seed: int
    checkpoint_key: str
    url: str
    summary: dict[str, Any]
    score: float

    def to_row(self) -> dict[str, Any]:
        row = {
            'seed': self.seed,
            'wandb url': self.url,
            'score': self.score,
        }
        return row


def results_to_dataframe(results: Sequence[Result]):
    assert results, 'No results to summarize'
    any_run = results[0]

    summary_cols = [
        k
        for k in any_run.summary.keys()  #
        if k == '_runtime' or k.startswith('labels/n') or k.startswith('val_')
    ]
    result_cols = list(any_run.to_row().keys())

    df = pd.DataFrame(
        [
            tuple(r.to_row().values()) + tuple(r.summary.get(k, None) for k in summary_cols)  #
            for r in results
        ],
        columns=result_cols + summary_cols,
    )

    return df


def pareto_front(df: pd.DataFrame, minimize: list[str], maximize: list[str]) -> pd.DataFrame:
    """
    Return Pareto-optimal (non-dominated) rows from a DataFrame.

    A solution dominates another if it's better-or-equal in all objectives
    and strictly better in at least one.

    Warning: This is an O(n^2) algorithm and may be slow for large DataFrames.
    """
    n = len(df)
    dominated = np.zeros(n, dtype=bool)

    # Extract values once for efficiency
    min_vals = df[minimize].values if minimize else np.empty((n, 0))
    max_vals = df[maximize].values if maximize else np.empty((n, 0))

    for i in range(n):
        if dominated[i]:
            continue

        for j in range(n):
            if i == j or dominated[j]:
                continue

            # Does j dominate i?
            # j dominates i if: j ≤ i (minimize) AND j ≥ i (maximize) AND strict in ≥1
            min_leq = np.all(min_vals[j] <= min_vals[i])
            max_geq = np.all(max_vals[j] >= max_vals[i])

            if min_leq and max_geq:
                min_strict = np.any(min_vals[j] < min_vals[i])
                max_strict = np.any(max_vals[j] > max_vals[i])

                if min_strict or max_strict:
                    dominated[i] = True
                    break

    return df[~dominated].copy()
