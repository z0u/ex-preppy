import logging
from dataclasses import dataclass
from typing import Any, override

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from ex_color.model import ColorMLPTrainingModule

log = logging.getLogger(__name__)


@dataclass
class MetricsRecord:
    """Simple metrics record without backward compatibility."""

    step: int
    total_loss: float
    losses: dict[str, float]


class MetricsCallback(Callback):
    """Simplified metrics callback using Lightning's built-in systems."""

    def __init__(self):
        super().__init__()
        self.history: list[MetricsRecord] = []

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record metrics after each training step."""
        del batch_idx
        # Extract loss information from outputs
        if isinstance(outputs, dict):
            total_loss = outputs.get('loss', 0.0)
            losses = outputs.get('losses', {})
        else:
            total_loss = outputs if outputs is not None else 0.0
            losses = {}

        if torch.is_tensor(total_loss):
            total_loss = total_loss.item()

        # Create metrics record
        record = MetricsRecord(step=trainer.global_step, total_loss=total_loss, losses=losses.copy())
        self.history.append(record)


class PhaseCallback(Callback):
    """Lightning callback to handle phase transitions."""

    @override
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Handle phase start events."""
        del batch_idx
        if not isinstance(pl_module, ColorMLPTrainingModule):
            return

        current_state = pl_module.timeline.state

        if current_state.is_phase_start:
            log.info(f'Starting phase: {current_state.phase}')


class ValidationCallback(Callback):
    """Lightning callback to handle validation at phase ends."""

    def __init__(self, val_data: torch.Tensor):
        super().__init__()
        self.val_data = val_data

    @override
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Handle phase end validation."""
        del batch_idx
        if not isinstance(pl_module, ColorMLPTrainingModule):
            return

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
