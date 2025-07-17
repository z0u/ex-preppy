import logging
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from ex_color.events import StepMetricsEvent
from ex_color.record import MetricsRecorder

log = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """Lightning callback to replace the metrics recording functionality."""

    def __init__(self):
        super().__init__()
        self.metrics_recorder = MetricsRecorder()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record metrics after each training step."""
        batch_data, batch_labels = batch

        # Extract loss information from outputs
        if isinstance(outputs, dict):
            total_loss = outputs.get('loss', 0.0)
        else:
            total_loss = outputs if outputs is not None else 0.0

        if torch.is_tensor(total_loss):
            total_loss = total_loss.item()

        # Get losses from logged metrics (if available)
        losses = {}
        if hasattr(trainer.logged_metrics, 'items'):
            for key, value in trainer.logged_metrics.items():
                if key.startswith('train_') and key != 'train_loss':
                    losses[key.replace('train_', '')] = value.item() if torch.is_tensor(value) else value

        # Create a proper StepMetricsEvent for the metrics recorder
        step_metrics_event = StepMetricsEvent(
            name='step-metrics',
            step=trainer.global_step,
            model=pl_module,
            timeline_state=pl_module.timeline.state,
            optimizer=trainer.optimizers[0] if trainer.optimizers else None,
            train_batch=batch_data,
            total_loss=total_loss,
            losses=losses,
        )

        self.metrics_recorder(step_metrics_event)


class PhaseCallback(Callback):
    """Lightning callback to handle phase transitions."""

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Handle phase start events."""
        current_state = pl_module.timeline.state

        if current_state.is_phase_start:
            log.info(f'Starting phase: {current_state.phase}')


class ValidationCallback(Callback):
    """Lightning callback to handle validation at phase ends."""

    def __init__(self, val_data: torch.Tensor):
        super().__init__()
        self.val_data = val_data

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Handle phase end validation."""
        current_state = pl_module.timeline.state

        if current_state.is_phase_end:
            log.info(f'Ending phase: {current_state.phase}')

            # Run validation
            pl_module.eval()
            with torch.no_grad():
                # Just run inference to trigger any validation effects
                _ = pl_module(self.val_data.to(pl_module.device))

                # Log validation metrics
                pl_module.log('val_step', trainer.global_step)

            pl_module.train()
