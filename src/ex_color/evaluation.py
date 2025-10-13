"""
Multi-seed evaluation framework for color experiments.

This module provides dataclasses and functions for running systematic evaluations
across multiple training seeds and intervention strategies.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import skimage as ski
import torch.nn as nn
from IPython.display import clear_output
from scipy.stats import pearsonr

from ex_color.data import hsv_similarity
from ex_color.data.color_cube import ColorCube
from ex_color.intervention.intervention import InterventionConfig

log = logging.getLogger(__name__)


@dataclass
class Resultset:
    """Results from evaluating a model with a specific intervention strategy."""

    tags: Sequence[str]
    latent_cube: ColorCube
    color_slice_cube: ColorCube
    loss_cube: ColorCube
    named_colors: Any  # pd.DataFrame


@dataclass(frozen=True)
class CorrelationStats:
    """Statistics from a correlation analysis between similarity and reconstruction error."""

    correlation: float
    r_squared: float
    p_value: float
    power: float

    def to_row(self, prefix: str) -> dict[str, float]:
        """Convert to a dictionary for DataFrame row insertion."""
        return {
            f'{prefix}_r': self.correlation,
            f'{prefix}_r2': self.r_squared,
            f'{prefix}_p': self.p_value,
            f'{prefix}_power': self.power,
        }


@dataclass(frozen=True)
class CorrelationSpec:
    """Specification for computing correlation between similarity and reconstruction error."""

    plan: str  # Name of the evaluation plan
    anchor_hsv: tuple[float, float, float]  # HSV color to compute similarity to
    power: float  # Power to raise similarity to (for emphasizing differences)


@dataclass
class EvaluationContext:
    """Context for evaluating a model with specific interventions."""

    model: nn.Module
    interventions: Sequence[InterventionConfig]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationPlan:
    """Plan for evaluating a model with a specific intervention strategy."""

    name: str
    tags: Sequence[str]
    setup: Callable[[nn.Module], EvaluationContext]


@dataclass(frozen=True)
class RunMetrics:
    """Metrics collected from a single training run."""

    seed: int
    correlations: Mapping[str, CorrelationStats]

    def score(self, plan: str) -> float:
        """
        Get the score for a specific evaluation plan.

        Uses absolute R² as the score, prioritizing strong correlations
        regardless of direction.
        """
        return abs(self.correlations[plan].r_squared)

    def to_row(self) -> dict[str, float | int]:
        """Convert to a dictionary for DataFrame row insertion."""
        row: dict[str, float | int] = {'seed': self.seed}
        for plan_name, stats in self.correlations.items():
            prefix = metric_prefix(plan_name)
            row.update(stats.to_row(prefix))
        return row


@dataclass
class BestRunArtifacts:
    """Complete artifacts from the best training run."""

    seed: int
    model: nn.Module
    metrics: RunMetrics
    plans: Mapping[str, EvaluationPlan]
    results: Mapping[str, Resultset]
    contexts: Mapping[str, EvaluationContext]
    named_colors: Any  # pd.DataFrame

    def get_extra(self, plan_name: str, key: str) -> Any:
        """Get an extra value from a specific evaluation context."""
        return self.contexts[plan_name].extras.get(key)


def metric_prefix(name: str) -> str:
    """Convert a plan name to a valid metric prefix for column names."""
    return name.replace(' ', '_').replace('-', '_')


def compute_similarity_to_anchor(
    cube: ColorCube,
    anchor_hsv: tuple[float, float, float],
    *,
    power: float = 1.0,
) -> ColorCube:
    """
    Compute similarity of each color in the cube to an anchor color.

    Args:
        cube: ColorCube with 'color' variable (RGB)
        anchor_hsv: Anchor color in HSV space
        power: Power to raise similarity to (for emphasis)

    Returns:
        ColorCube with added 'hsv' and 'similarity' variables
    """
    cube = cube.assign(hsv=ski.color.rgb2hsv(cube['color']))
    cube = cube.assign(similarity=hsv_similarity(cube['hsv'], np.array(anchor_hsv), hemi=True, mode='angular') ** power)
    return cube


def correlation_stats(
    cube: ColorCube,
    anchor_hsv: tuple[float, float, float],
    *,
    power: float = 1.0,
) -> CorrelationStats:
    """
    Compute correlation statistics between similarity and reconstruction error.

    Args:
        cube: ColorCube with 'MSE' variable
        anchor_hsv: Anchor color in HSV space to compute similarity to
        power: Power to raise similarity to

    Returns:
        CorrelationStats with correlation, R², p-value, and power
    """
    cube = compute_similarity_to_anchor(cube, anchor_hsv, power=power)
    similarity_flat = cube['similarity'].flatten()
    mse_flat = cube['MSE'].flatten()

    r, p = pearsonr(similarity_flat, mse_flat)
    return CorrelationStats(
        correlation=float(r),
        r_squared=float(r**2),
        p_value=float(p),
        power=power,
    )


def evaluate_single_plan(
    model: nn.Module,
    plan: EvaluationPlan,
    test_fn: Callable[[nn.Module, list[InterventionConfig], ColorCube], ColorCube],
    test_data: ColorCube,
) -> tuple[Resultset, EvaluationContext]:
    """
    Evaluate a model with a single intervention plan.

    Args:
        model: Trained model to evaluate
        plan: Evaluation plan specifying the intervention strategy
        test_fn: Function to run inference with interventions
        test_data: ColorCube to test on

    Returns:
        Tuple of (Resultset, EvaluationContext)
    """
    log.debug(f'Evaluating plan: {plan.name}')
    context = plan.setup(model)

    # Run inference with interventions
    result = test_fn(context.model, list(context.interventions), test_data)

    resultset = Resultset(
        tags=plan.tags,
        latent_cube=result,
        color_slice_cube=result.permute('svh'),
        loss_cube=result,
        named_colors=None,  # Filled in separately if needed
    )

    return resultset, context


async def evaluate_seed(
    seed: int,
    model_factory: Callable[[int], Awaitable[nn.Module]],
    plans: Mapping[str, EvaluationPlan],
    correlation_specs: Sequence[CorrelationSpec],
    test_fn: Callable[[nn.Module, list[InterventionConfig], ColorCube], ColorCube],
    test_data: ColorCube,
) -> tuple[RunMetrics, nn.Module, dict[str, EvaluationContext]]:
    """
    Train and evaluate a model with a specific seed.

    Args:
        seed: Random seed for this run
        model_factory: Function that takes a seed and returns a trained model
        plans: Dictionary of evaluation plans by name
        correlation_specs: Specifications for computing correlations
        test_fn: Function to run inference with interventions
        test_data: ColorCube to test on

    Returns:
        Tuple of (RunMetrics, trained model, evaluation contexts)
    """
    # Train model
    model = await model_factory(seed)

    # Evaluate with each plan
    contexts: dict[str, EvaluationContext] = {}
    results: dict[str, Resultset] = {}

    for plan in plans.values():
        result, context = evaluate_single_plan(model, plan, test_fn, test_data)
        results[plan.name] = result
        contexts[plan.name] = context

    # Compute correlations for specified plans
    correlations: dict[str, CorrelationStats] = {}
    for spec in correlation_specs:
        correlations[spec.plan] = correlation_stats(
            results[spec.plan].latent_cube,
            anchor_hsv=spec.anchor_hsv,
            power=spec.power,
        )

    metrics = RunMetrics(seed=seed, correlations=correlations)
    return metrics, model, contexts


async def collect_full_results(
    seed: int,
    model: nn.Module,
    plans: Sequence[EvaluationPlan],
    test_fn: Callable[[nn.Module, list[InterventionConfig], ColorCube], ColorCube],
    test_fn_named: Callable[[nn.Module, list[InterventionConfig], Any], Any],
    test_data: ColorCube,
    named_colors_factory: Callable[[], Any],
    correlation_specs: Sequence[CorrelationSpec],
    *,
    precomputed_contexts: dict[str, EvaluationContext] | None = None,
) -> BestRunArtifacts:
    """
    Collect complete evaluation results for the best run.

    Args:
        seed: Random seed used for this run
        model: Trained model
        plans: List of evaluation plans
        test_fn: Function to run inference on ColorCube
        test_fn_named: Function to run inference on named colors
        test_data: ColorCube to test on
        named_colors_factory: Function to create named colors DataFrame
        correlation_specs: Specifications for computing correlations
        precomputed_contexts: Optional pre-computed evaluation contexts

    Returns:
        BestRunArtifacts with complete results
    """
    base_named_colors = named_colors_factory()

    results: dict[str, Resultset] = {}
    contexts: dict[str, EvaluationContext] = precomputed_contexts or {}

    for plan in plans:
        if plan.name in contexts:
            context = contexts[plan.name]
        else:
            context = plan.setup(model)
            contexts[plan.name] = context

        result = test_fn(context.model, list(context.interventions), test_data)
        named_result = test_fn_named(context.model, list(context.interventions), base_named_colors)

        results[plan.name] = Resultset(
            tags=plan.tags,
            latent_cube=result,
            color_slice_cube=result.permute('svh'),
            loss_cube=result,
            named_colors=named_result,
        )

    correlations: dict[str, CorrelationStats] = {}
    for spec in correlation_specs:
        correlations[spec.plan] = correlation_stats(
            results[spec.plan].latent_cube,
            anchor_hsv=spec.anchor_hsv,
            power=spec.power,
        )

    metrics = RunMetrics(seed=seed, correlations=correlations)
    plans_by_name = {plan.name: plan for plan in plans}

    return BestRunArtifacts(
        seed=seed,
        model=model,
        metrics=metrics,
        plans=plans_by_name,
        results=results,
        contexts=contexts,
        named_colors=base_named_colors,
    )


async def run_multi_seed_training(
    seeds: Sequence[int],
    train_fn: Callable[[int], Awaitable[nn.Module]],
    plans: Sequence[EvaluationPlan],
    correlation_specs: Sequence[CorrelationSpec],
    test_fn: Callable[[nn.Module, list[InterventionConfig], ColorCube], ColorCube],
    test_fn_named: Callable[[nn.Module, list[InterventionConfig], Any], Any],
    test_data: ColorCube,
    named_colors_factory: Callable[[], Any],
    *,
    best_plan: str,
) -> tuple[list[RunMetrics], BestRunArtifacts]:
    """
    Run training with multiple seeds and return metrics and best run artifacts.

    Args:
        seeds: List of random seeds to use
        train_fn: Function that takes a seed and returns a trained model
        plans: List of evaluation plans
        correlation_specs: Specifications for computing correlations
        test_fn: Function to run inference on ColorCube
        test_fn_named: Function to run inference on named colors
        test_data: ColorCube to test on
        named_colors_factory: Function to create named colors DataFrame
        best_plan: Name of the plan to use for selecting the best run

    Returns:
        Tuple of (list of RunMetrics, BestRunArtifacts)
    """
    metrics: list[RunMetrics] = []
    best: tuple[RunMetrics, nn.Module, dict[str, EvaluationContext]] | None = None
    plans_by_name = {plan.name: plan for plan in plans}

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        clear_output()
        print(f'Best so far: {best[0].to_row() if best is not None else "N/A"}')
        print(f'[{index}/{total_runs}] Training seed {seed}...')

        run_metrics, model, contexts = await evaluate_seed(
            seed,
            train_fn,
            plans_by_name,
            correlation_specs,
            test_fn,
            test_data,
        )
        metrics.append(run_metrics)

        if best is None or run_metrics.score(best_plan) > best[0].score(best_plan):
            best = (run_metrics, model, contexts)
        else:
            del model

    assert best is not None
    best_metrics, best_model, best_contexts = best

    best_artifacts = await collect_full_results(
        best_metrics.seed,
        best_model,
        plans,
        test_fn,
        test_fn_named,
        test_data,
        named_colors_factory,
        correlation_specs,
        precomputed_contexts=best_contexts,
    )

    return metrics, best_artifacts
