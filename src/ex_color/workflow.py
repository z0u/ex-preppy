"""
Reusable training and inference workflows for color experiments.

This module provides high-level functions that encapsulate common patterns found
across multiple experiment notebooks.
"""

import logging
from tempfile import gettempdir
from typing import Any, Sequence

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ex_color.callbacks import LabelProportionCallback
from ex_color.inference import InferenceModule
from ex_color.intervention.intervention import InterventionConfig
from ex_color.loss.regularizer import RegularizerConfig
from ex_color.model import CNColorMLP
from ex_color.training import TrainingModule
from mini.temporal.dopesheet import Dopesheet
from utils.progress.lightning import LightningProgress

log = logging.getLogger(__name__)


def train_model(
    model: CNColorMLP,
    dopesheet: Dopesheet,
    regularizers: list[RegularizerConfig],
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    experiment_name: str,
    project: str,
    hparams: dict[str, Any] | None = None,
) -> CNColorMLP:
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
        hparams: Optional hyperparameters to log

    Returns:
        Trained model
    """
    import wandb

    log.info(f'Training with: {[r.name for r in regularizers]}')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.debug(f'Training model with {total_params:,} trainable parameters.')

    training_module = TrainingModule(model, dopesheet, torch.nn.MSELoss(), regularizers)

    logger = WandbLogger(experiment_name, project=project, save_dir=gettempdir())
    logger.log_hyperparams(hparams or {})

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
    interventions: Sequence[InterventionConfig],
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
