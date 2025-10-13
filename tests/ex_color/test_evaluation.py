"""Tests for the evaluation module."""

import numpy as np
import pytest
import torch.nn as nn
from pytest import approx

from ex_color.data.color_cube import ColorCube
from ex_color.evaluation import (
    BestRunArtifacts,
    CorrelationStats,
    EvaluationContext,
    EvaluationPlan,
    Resultset,
    RunMetrics,
    compute_similarity_to_anchor,
    correlation_stats,
    metric_prefix,
)


@pytest.fixture
def simple_color_cube():
    """Create a simple color cube for testing."""
    return ColorCube.from_rgb(
        r=np.linspace(0, 1, 3),
        g=np.linspace(0, 1, 3),
        b=np.linspace(0, 1, 3),
    )


def test_metric_prefix():
    """Test metric_prefix converts plan names to valid prefixes."""
    assert metric_prefix('no intervention') == 'no_intervention'
    assert metric_prefix('ablated') == 'ablated'
    assert metric_prefix('some-plan') == 'some_plan'
    assert metric_prefix('complex plan-name') == 'complex_plan_name'


def test_correlation_stats_to_row():
    """Test CorrelationStats.to_row converts to dict correctly."""
    stats = CorrelationStats(correlation=0.5, r_squared=0.25, p_value=0.01, power=2.0)
    row = stats.to_row('test')

    assert row == approx({'test_r': 0.5, 'test_r2': 0.25, 'test_p': 0.01, 'test_power': 2.0})


def test_run_metrics_score():
    """Test RunMetrics.score returns absolute R²."""
    stats = CorrelationStats(correlation=-0.7, r_squared=0.49, p_value=0.01, power=2.0)
    metrics = RunMetrics(seed=42, correlations={'plan': stats})

    assert metrics.score('plan') == approx(0.49)


def test_run_metrics_to_row():
    """Test RunMetrics.to_row converts to dict correctly."""
    stats = CorrelationStats(correlation=0.5, r_squared=0.25, p_value=0.01, power=2.0)
    metrics = RunMetrics(seed=42, correlations={'plan': stats})
    row = metrics.to_row()

    assert row['seed'] == 42
    assert row['plan_r'] == approx(0.5)
    assert row['plan_r2'] == approx(0.25)
    assert row['plan_p'] == approx(0.01)
    assert row['plan_power'] == approx(2.0)


def test_compute_similarity_to_anchor(simple_color_cube):
    """Test compute_similarity_to_anchor adds similarity variable."""
    anchor_hsv = (0.0, 1.0, 1.0)  # Pure red
    result = compute_similarity_to_anchor(simple_color_cube, anchor_hsv, power=1.0)

    assert 'hsv' in result.vars
    assert 'similarity' in result.vars
    assert result['similarity'].shape == simple_color_cube.shape


def test_correlation_stats(simple_color_cube):
    """Test correlation_stats computes correlations correctly."""
    # Add some MSE data that correlates with red similarity
    simple_color_cube = simple_color_cube.assign(
        MSE=simple_color_cube['color'][..., 0]  # Use red channel as proxy for MSE
    )

    anchor_hsv = (0.0, 1.0, 1.0)  # Pure red
    stats = correlation_stats(simple_color_cube, anchor_hsv, power=1.0)

    assert isinstance(stats, CorrelationStats)
    assert isinstance(stats.correlation, float)
    assert isinstance(stats.r_squared, float)
    assert isinstance(stats.p_value, float)
    assert stats.power == 1.0
    # Correlation should exist (not NaN)
    assert not np.isnan(stats.correlation)


def test_evaluation_context():
    """Test EvaluationContext dataclass."""
    model = nn.Linear(3, 3)
    interventions = []
    extras = {'key': 'value'}

    context = EvaluationContext(model=model, interventions=interventions, extras=extras)

    assert context.model is model
    assert context.interventions == interventions
    assert context.extras == extras


def test_evaluation_plan():
    """Test EvaluationPlan dataclass."""
    model = nn.Linear(3, 3)

    def setup_fn(m):
        return EvaluationContext(model=m, interventions=[])

    plan = EvaluationPlan(name='test', tags=['tag1', 'tag2'], setup=setup_fn)

    assert plan.name == 'test'
    assert plan.tags == ['tag1', 'tag2']

    context = plan.setup(model)
    assert isinstance(context, EvaluationContext)
    assert context.model is model


def test_resultset():
    """Test Resultset dataclass."""
    cube = ColorCube.from_rgb(r=np.array([0, 1]), g=np.array([0, 1]), b=np.array([0, 1]))

    resultset = Resultset(
        tags=['test'],
        latent_cube=cube,
        color_slice_cube=cube,
        loss_cube=cube,
        named_colors=None,
    )

    assert resultset.tags == ['test']
    assert resultset.latent_cube is cube
    assert resultset.color_slice_cube is cube
    assert resultset.loss_cube is cube


def test_best_run_artifacts():
    """Test BestRunArtifacts dataclass."""
    model = nn.Linear(3, 3)
    stats = CorrelationStats(correlation=0.5, r_squared=0.25, p_value=0.01, power=2.0)
    metrics = RunMetrics(seed=42, correlations={'plan': stats})
    context = EvaluationContext(model=model, interventions=[], extras={'key': 'value'})
    cube = ColorCube.from_rgb(r=np.array([0, 1]), g=np.array([0, 1]), b=np.array([0, 1]))
    resultset = Resultset(
        tags=['test'],
        latent_cube=cube,
        color_slice_cube=cube,
        loss_cube=cube,
        named_colors=None,
    )

    def setup_fn(m):
        return context

    plan = EvaluationPlan(name='plan', tags=['test'], setup=setup_fn)

    artifacts = BestRunArtifacts(
        seed=42,
        model=model,
        metrics=metrics,
        plans={'plan': plan},
        results={'plan': resultset},
        contexts={'plan': context},
        named_colors=None,
    )

    assert artifacts.seed == 42
    assert artifacts.model is model
    assert artifacts.metrics is metrics
    assert artifacts.get_extra('plan', 'key') == 'value'
    assert artifacts.get_extra('plan', 'missing') is None
