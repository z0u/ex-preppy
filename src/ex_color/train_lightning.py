import logging

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from ex_color.criteria.criteria import LossCriterion, RegularizerConfig
from ex_color.lightning_callbacks import MetricsCallback, PhaseCallback, ValidationCallback
from ex_color.model import ColorMLP
from mini.temporal.dopesheet import Dopesheet

log = logging.getLogger(__name__)


def train_color_model_lightning(
    train_loader: DataLoader,
    val_data: torch.Tensor,
    dopesheet: Dopesheet,
    loss_criterion: LossCriterion,
    regularizers: list[RegularizerConfig],
) -> MetricsCallback:
    """Train the color model using PyTorch Lightning."""
    # Create the Lightning module
    model = ColorMLP(dopesheet, loss_criterion, regularizers)

    # Set up callbacks
    metrics_callback = MetricsCallback()
    phase_callback = PhaseCallback()
    validation_callback = ValidationCallback(val_data)
    progress_bar = TQDMProgressBar()

    callbacks = [
        metrics_callback,
        phase_callback,
        validation_callback,
        progress_bar,
    ]

    # Create trainer with correct number of steps
    from mini.temporal.timeline import Timeline
    total_steps = len(Timeline(dopesheet))
    trainer = L.Trainer(
        max_steps=total_steps,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=False,  # Disable logging to avoid conflicts
    )

    # Train the model
    trainer.fit(model, train_loader)

    return metrics_callback
