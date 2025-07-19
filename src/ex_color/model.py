import logging
from typing import Protocol, override

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from ex_color.regularizers.regularizer import RegularizerConfig
from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline

E = 4

log = logging.getLogger(__name__)


class ColorMLP(nn.Module):
    """Pure neural network model for color transformation."""

    def __init__(self):
        super().__init__()
        # RGB input (3D) → hidden layer → bottleneck → hidden layer → RGB output
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, E),  # Our critical bottleneck!
        )

        self.decoder = nn.Sequential(
            nn.Linear(E, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),  # Keep RGB values in [0,1]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Get our bottleneck representation
        latents = self.encoder(x)

        # Decode back to RGB
        return self.decoder(latents)


class LatentCaptureHook:
    """Hook class to capture latent representations externally."""

    def __init__(self):
        self.current_latents: Tensor | None = None

    def __call__(self, _module, _input, output):
        """Hook function to capture latent representations."""
        self.current_latents = output


class Objective(Protocol):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor: ...


class ColorMLPTrainingModule(L.LightningModule):
    """Lightning module that handles training logic."""

    def __init__(
        self,
        dopesheet: Dopesheet,
        objective: Objective,
        regularizers: list[RegularizerConfig],
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store training configuration
        self.objective = objective
        self.regularizers = regularizers
        self.timeline = Timeline(dopesheet)

        # Create the pure neural network model (without training logic)
        self.model = ColorMLP()

        # Set up external latent capture hook
        # TODO: How to regularize the activations of layers in a deep network, without hardcoding it here? Maybe bind regularizers to layers with self.model.get_submodule('some.module')
        self.latent_hook = LatentCaptureHook()
        self.model.encoder.register_forward_hook(self.latent_hook)

    def on_save_checkpoint(self, checkpoint):
        log.info('Saving timeline state')
        checkpoint['timeline_step'] = self.timeline._step

    def on_load_checkpoint(self, checkpoint):
        if 'timeline_step' in checkpoint:
            log.info('Restoring timeline state')
            # Fast-forward timeline to saved position
            for _ in range(checkpoint['timeline_step']):
                self.timeline.step()

    @property
    def schedule(self) -> dict[str, float]:
        return self.timeline.state.props

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @override
    def training_step(self, batch: tuple[Tensor, dict[str, Tensor]], batch_idx):
        batch_data, batch_labels = batch
        assert all(isinstance(k, str) and isinstance(v, Tensor) for k, v in batch_labels.items())

        # Forward pass
        outputs: Tensor = self(batch_data)
        assert self.latent_hook.current_latents is not None

        # Calculate primary reconstruction loss
        primary_loss = self.objective(outputs, batch_data).mean()
        losses = {'recon': primary_loss.item()}
        total_loss = primary_loss

        # Calculate regularizer losses
        for reg in self.regularizers:
            # Default to full weight for regularizers that aren't in the timeline
            weight = self.schedule.get(reg.name, 1.0)
            if weight == 0:
                # Regularizer has been disabled for this timestep
                continue

            term_loss = compute_loss_term(reg, batch_labels, self.latent_hook.current_latents)
            if term_loss is None:
                continue

            if not torch.isfinite(term_loss):
                raise RuntimeError(f'Loss term {reg.name} at step {self.global_step} is not finite: {term_loss}')

            losses[reg.name] = term_loss.item()
            total_loss += term_loss * weight

        # Log metrics using Lightning's built-in logging
        self.log('train_loss', total_loss, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value)

        return {'loss': total_loss, 'losses': losses}

    @override
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-8)  # Initial LR will be overridden

    @override
    def on_before_optimizer_step(self, optimizer):
        current_lr = self.schedule.get('lr', 1e-8)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    @override
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step < len(self.timeline) - 1:
            self.timeline.step()


def compute_loss_term(regularizer: RegularizerConfig, batch_labels, latents: Tensor):
    per_sample_loss = regularizer.compute_loss_term(latents)

    if regularizer.label_affinities is None:
        return regularizer.compute_loss_term(latents).mean()

    sample_affinities = get_sample_affinities(batch_labels, regularizer.label_affinities)
    if sample_affinities is None:
        # No labels match
        return None

    # Weight per-sample loss by the regularizer's affinity for its labels
    if len(per_sample_loss.shape) == 0:
        # If the loss is a scalar, we need to expand it to match the batch size
        per_sample_loss = per_sample_loss.expand(sample_affinities.shape[0])
    elif per_sample_loss.shape[0] != sample_affinities.shape[0]:
        raise RuntimeError(f'Loss should be per-sample OR scalar: {regularizer.name}')
    weighted_loss = per_sample_loss * sample_affinities
    return weighted_loss.sum() / (sample_affinities.sum() + 1e-8)


def get_sample_affinities(batch_labels: dict[str, Tensor], label_affinities: dict[str, float]):
    # Soft labels that indicate how much effect this regularizer has (in practice, labels may be binary)
    label_probs = [
        batch_labels[k] * v  #
        for k, v in label_affinities.items()
        if k in batch_labels
    ]
    if not label_probs:
        return None

    sample_affinities = torch.stack(label_probs, dim=0).sum(dim=0)
    sample_affinities = torch.clamp(sample_affinities, 0.0, 1.0)
    return sample_affinities
