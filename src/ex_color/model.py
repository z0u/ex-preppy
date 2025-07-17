import logging

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from ex_color.criteria.criteria import LossCriterion, RegularizerConfig
from ex_color.result import InferenceResult
from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline

E = 4

log = logging.getLogger(__name__)


class ColorMLP(L.LightningModule):
    def __init__(
        self,
        dopesheet: Dopesheet,
        loss_criterion: LossCriterion,
        regularizers: list[RegularizerConfig],
    ):
        super().__init__()
        # Store configuration
        self.loss_criterion = loss_criterion
        self.regularizers = regularizers
        self.timeline = Timeline(dopesheet)

        # Store latents for regularization via hooks
        self.current_latents: Tensor | None = None

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

        # Register forward hook to capture latents
        self.encoder.register_forward_hook(self._capture_latents_hook)

    def _capture_latents_hook(self, module, input, output):
        """Hook to capture latent representations from the encoder."""
        self.current_latents = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the reconstructed RGB values."""
        # Get our bottleneck representation
        bottleneck = self.encoder(x)

        # Decode back to RGB
        output = self.decoder(bottleneck)
        return output

    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        batch_data, batch_labels = batch

        # Get current timeline state
        current_state = self.timeline.state

        # Forward pass
        outputs = self(batch_data)
        current_results = InferenceResult(outputs, self.current_latents)

        # Calculate primary reconstruction loss
        primary_loss = self.loss_criterion(batch_data, current_results).mean()
        losses = {'recon': primary_loss.item()}
        total_loss = primary_loss
        zeros = torch.tensor(0.0, device=batch_data.device)

        # Calculate regularizer losses
        for regularizer in self.regularizers:
            name = regularizer.name
            criterion = regularizer.criterion

            weight = current_state.props.get(name, 1.0)
            if weight == 0:
                continue

            if regularizer.label_affinities is not None:
                # Soft labels that indicate how much effect this regularizer has
                label_probs = [
                    batch_labels[k] * v
                    for k, v in regularizer.label_affinities.items()
                    if k in batch_labels
                ]
                if not label_probs:
                    continue

                sample_affinities = torch.stack(label_probs, dim=0).sum(dim=0)
                sample_affinities = torch.clamp(sample_affinities, 0.0, 1.0)
                if torch.allclose(sample_affinities, zeros):
                    continue
            else:
                sample_affinities = torch.ones(batch_data.shape[0], device=batch_data.device)

            per_sample_loss = criterion(batch_data, current_results)
            if len(per_sample_loss.shape) == 0:
                # If the loss is a scalar, we need to expand it to match the batch size
                per_sample_loss = per_sample_loss.expand(batch_data.shape[0])
            assert per_sample_loss.shape[0] == batch_data.shape[0], f'Loss should be per-sample OR scalar: {name}'

            # Apply sample affinities
            weighted_loss = per_sample_loss * sample_affinities

            # Calculate mean only over selected samples
            term_loss = weighted_loss.sum() / (sample_affinities.sum() + 1e-8)

            losses[name] = term_loss.item()
            if not torch.isfinite(term_loss):
                log.warning(f'Loss term {name} at step {self.global_step} is not finite: {term_loss}')
                continue
            total_loss += term_loss * weight

        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value)

        return total_loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.Adam(self.parameters(), lr=1e-8)  # Initial LR will be overridden
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        """Update learning rate based on timeline before optimizer step."""
        current_state = self.timeline.state
        current_lr = current_state.props.get('lr', 1e-8)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Advance timeline after each training step."""
        if self.global_step < len(self.timeline) - 1:
            self.timeline.step()
