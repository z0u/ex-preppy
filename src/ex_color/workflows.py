"""
Reusable training and inference workflows for color experiments.

This module provides high-level functions that encapsulate common patterns found
across multiple experiment notebooks.
"""

import logging
from tempfile import gettempdir

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ex_color.callbacks import LabelProportionCallback
from ex_color.data.color_cube import ColorCube
from ex_color.inference import InferenceModule
from ex_color.intervention.intervention import InterventionConfig
from ex_color.loss.regularizer import RegularizerConfig
from ex_color.seed import set_deterministic_mode
from ex_color.training import TrainingModule
from mini.temporal.dopesheet import Dopesheet
from torch.nn import functional as F
from utils.progress.lightning import LightningProgress

log = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    dopesheet: Dopesheet,
    regularizers: list[RegularizerConfig],
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str,
    project: str,
    seed: int | None = None,
) -> nn.Module:
    """
    Train a model with the given dopesheet and regularizers.

    Args:
        model: PyTorch model to train
        dopesheet: Training schedule configuration
        regularizers: List of regularizer configurations
        train_loader: Training data loader
        val_loader: Validation data loader
        experiment_name: Name for the experiment (used in logging)
        project: Project name for Weights & Biases
        seed: Optional random seed for reproducibility

    Returns:
        Trained model
    """
    import wandb

    log.info(f'Training with: {[r.name for r in regularizers]}')

    if seed is not None:
        set_deterministic_mode(seed)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.debug(f'Model initialized with {total_params:,} trainable parameters.')

    training_module = TrainingModule(model, dopesheet, torch.nn.MSELoss(), regularizers)

    logger = WandbLogger(experiment_name, project=project, save_dir=gettempdir())
    logger.log_hyperparams({'seed': seed})

    trainer = L.Trainer(
        max_steps=len(dopesheet),
        callbacks=[
            LightningProgress(),
            LabelProportionCallback(
                prefix='labels',
                get_active_labels=lambda: training_module.active_labels,
            ),
        ],
        enable_checkpointing=False,
        enable_model_summary=False,
        val_check_interval=len(dopesheet) // 10,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=min(50, len(train_loader)),
    )

    print(f'max_steps: {len(dopesheet)}, train_loader length: {len(train_loader)}')

    # Train the model
    try:
        trainer.fit(training_module, train_loader, val_loader)
    finally:
        wandb.finish()

    return model


def infer_with_latent_capture(
    model: nn.Module,
    test_data: Tensor,
    interventions: list[InterventionConfig],
    layer_name: str = 'bottleneck',
    num_workers: int = 0,
) -> tuple[Tensor, Tensor]:
    """
    Run inference on test data and capture latent activations.

    Args:
        model: Trained model
        test_data: Test data tensor with shape [..., 3] (RGB)
        interventions: List of intervention configurations to apply
        layer_name: Name of the layer to capture activations from
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (predictions, latents):
        - predictions: Model outputs with same shape as test_data
        - latents: Captured latent activations, shape [N, D] where N is total samples
    """
    module = InferenceModule(model, interventions, capture_layers=[layer_name])

    trainer = L.Trainer(
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    batches = trainer.predict(
        module,
        DataLoader(
            TensorDataset(test_data.reshape((-1, 3))),
            batch_size=64,
            collate_fn=lambda batch: torch.stack([row[0] for row in batch], 0),
            num_workers=num_workers,
        ),
    )
    assert batches is not None
    preds = [item for batch in batches for item in batch]
    y = torch.cat(preds).reshape(test_data.shape)
    # Read captured activations as a flat [N, D] tensor
    latents = module.read_captured(layer_name)
    return y, latents


def evaluate_model_on_cube(
    model: nn.Module,
    interventions: list[InterventionConfig],
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
    interventions: list[InterventionConfig],
    test_data,  # pd.DataFrame
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
    y_tuples = [tuple(row) for row in y.numpy()]
    return test_data.assign(recon=y_tuples, MSE=per_color_loss.numpy())
