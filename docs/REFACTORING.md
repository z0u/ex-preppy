# Refactored Modules Guide

This document describes the refactored modules extracted from the ex-2.9.1-redux notebook, which provide reusable patterns for color experiments.

## Overview

Three new modules have been created to promote code reuse across experiment notebooks:

1. **`ex_color.workflows`** - Training and inference workflows
2. **`ex_color.evaluation`** - Multi-seed evaluation framework
3. **`ex_color.vis.helpers`** - Common visualization patterns

## Workflows Module (`ex_color.workflows`)

### Training

```python
from ex_color.workflows import train_model
from ex_color.model import CNColorMLP
from mini.temporal.dopesheet import Dopesheet

# Create model
model = CNColorMLP(k_bottleneck=5, n_nonlinear=2, k_codec=10)

# Load dopesheet
dopesheet = Dopesheet.from_csv('./ex-2.9.1-dopesheet.csv')

# Train
trained_model = train_model(
    model=model,
    dopesheet=dopesheet,
    regularizers=[reg_separate, reg_anchor, reg_anti_subspace],
    train_loader=train_loader,
    val_loader=val_loader,
    experiment_name='My Experiment',
    project='ex-preppy',
    seed=42,
)
```

### Inference and Evaluation

```python
from ex_color.workflows import (
    evaluate_model_on_cube,
    evaluate_model_on_named_colors,
    infer_with_latent_capture,
)
from ex_color.data.color_cube import ColorCube
import numpy as np

# Create test data
test_cube = ColorCube.from_rgb(
    r=np.linspace(0, 1, 20),
    g=np.linspace(0, 1, 20),
    b=np.linspace(0, 1, 20),
)

# Evaluate on cube (returns cube with 'recon', 'MSE', 'latents')
result = evaluate_model_on_cube(
    model=trained_model,
    interventions=[],  # or your intervention configs
    test_data=test_cube,
)

# Access results
print(f"Max MSE: {np.max(result['MSE'])}")
print(f"Latent shape: {result['latents'].shape}")

# Or just capture latents during inference
predictions, latents = infer_with_latent_capture(
    model=trained_model,
    test_data=torch.tensor(test_cube.rgb_grid, dtype=torch.float32),
    interventions=[],
    layer_name='bottleneck',
)
```

## Evaluation Module (`ex_color.evaluation`)

### Multi-Seed Training and Evaluation

```python
from ex_color.evaluation import (
    EvaluationPlan,
    EvaluationContext,
    CorrelationSpec,
    run_multi_seed_training,
)
from ex_color.surgery import ablate, prune
from ex_color.intervention import Suppression, BoundedFalloff

# Define evaluation plans
def no_intervention_plan(model):
    return EvaluationContext(model=model, interventions=())

def ablation_plan(model):
    ablated = ablate(model, 'bottleneck', [0])
    return EvaluationContext(model=ablated, interventions=())

EVALUATION_PLANS = [
    EvaluationPlan(name='baseline', tags=['no intervention'], setup=no_intervention_plan),
    EvaluationPlan(name='ablated', tags=['ablated'], setup=ablation_plan),
]

# Define correlation specs
ANCHOR_HSV = (0.0, 1.0, 1.0)  # Pure red
CORRELATION_SPECS = [
    CorrelationSpec(plan='ablated', anchor_hsv=ANCHOR_HSV, power=3),
]

# Training function (async)
async def train_single_seed(seed):
    # ... your training code here ...
    return trained_model

# Run multi-seed training
run_metrics, best_run = await run_multi_seed_training(
    seeds=range(10),  # Try 10 different seeds
    train_fn=train_single_seed,
    plans=EVALUATION_PLANS,
    correlation_specs=CORRELATION_SPECS,
    test_fn=evaluate_model_on_cube,
    test_fn_named=evaluate_model_on_named_colors,
    test_data=test_cube,
    named_colors_factory=lambda: get_named_colors_df(n_hues=12, n_grays=5),
    best_plan='ablated',  # Select best by this plan's R²
)

# Access best run artifacts
print(f"Best seed: {best_run.seed}")
print(f"Best R²: {best_run.metrics.correlations['ablated'].r_squared}")

# Access results for specific plans
ablation_results = best_run.results['ablated']
print(f"Max loss: {np.max(ablation_results.loss_cube['MSE'])}")
```

### Working with Results

```python
from ex_color.evaluation import correlation_stats, compute_similarity_to_anchor

# Compute correlation between similarity and reconstruction error
stats = correlation_stats(
    cube=result_cube,
    anchor_hsv=(0.0, 1.0, 1.0),  # Pure red
    power=2.0,
)
print(f"Correlation: {stats.correlation:.3f}")
print(f"R²: {stats.r_squared:.3f}")
print(f"p-value: {stats.p_value:.4f}")

# Add similarity to a cube
cube_with_sim = compute_similarity_to_anchor(
    cube=test_cube,
    anchor_hsv=(0.0, 1.0, 1.0),
    power=2.0,
)
# Now cube_with_sim has 'hsv' and 'similarity' variables
```

## Visualization Helpers Module (`ex_color.vis.helpers`)

### Common Visualization Functions

```python
from ex_color.vis.helpers import (
    visualize_reconstructed_cube,
    visualize_reconstruction_loss,
    visualize_stacked_results,
    visualize_error_vs_similarity,
    hstack_named_results,
)

# Visualize reconstructions
visualize_reconstructed_cube(
    data=result_cube,
    tags=['ablated', 'seed-42'],
    nbid='2.9.1',
)

# Visualize loss patterns
visualize_reconstruction_loss(
    data=result_cube,
    tags=['ablated'],
    nbid='2.9.1',
)

# Stacked visualization (latents + colors + loss)
visualize_stacked_results(
    resultset=ablation_results,
    latent_dims=((3, 0, 1), (3, 2, 4)),
    max_error=0.1,
    nbid='2.9.1',
)

# Compare multiple results in a table
comparison_df = hstack_named_results(
    baseline_results,
    ablation_results,
    suppression_results,
)
display(comparison_df)
```

## Migration from Notebook Code

If you have existing notebook code, here's how to migrate:
