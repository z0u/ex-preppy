import logging

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from ex_color.lightning_callbacks import MetricsCallback, PhaseCallback, ValidationCallback
from ex_color.model import Objective, TrainingModule
from ex_color.regularizers.regularizer import RegularizerConfig
from mini.temporal.dopesheet import Dopesheet

log = logging.getLogger(__name__)


def train_color_model_lightning(
    train_loader: DataLoader,
    val_data: torch.Tensor,
    dopesheet: Dopesheet,
    objective: Objective,
    regularizers: list[RegularizerConfig],
    model: torch.nn.Module,
) -> MetricsCallback:
    """
    Train the color model using PyTorch Lightning.

    Args:
        train_loader: DataLoader for training data
        val_data: Validation data
        dopesheet: Training schedule
        objective: Loss function
        regularizers: List of regularizers to apply
        model: PyTorch model to use
    """
    training_module = TrainingModule(model, dopesheet, objective, regularizers)

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

    trainer = L.Trainer(
        max_steps=len(dopesheet),
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=False,  # Disable logging to avoid conflicts
    )

    # Train the model
    trainer.fit(training_module, train_loader)

    return metrics_callback
