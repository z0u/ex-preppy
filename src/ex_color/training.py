import logging
from typing import cast, override

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from ex_color.hooks import ActivationCaptureHook
from ex_color.loss.objective import Objective
from ex_color.loss.regularizer import RegularizerConfig
from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline

log = logging.getLogger(__name__)


class TrainingModule(L.LightningModule):
    """Lightning module that handles training logic for any model architecture."""

    def __init__(
        self,
        model: nn.Module,
        dopesheet: Dopesheet,
        objective: Objective,
        regularizers: list[RegularizerConfig],
    ):
        super().__init__()

        # Save params for rehydration. Modules are saved automatically.
        ignore = ['model']
        if isinstance(objective, nn.Module):
            ignore.append('objective')
            objective = cast(Objective, objective)  # Consider it to be a module that implements the Objective protocol
        self.save_hyperparameters({'dopesheet': dopesheet.to_dict()})
        self.save_hyperparameters(ignore=ignore + ['dopesheet'])

        # Store training configuration
        self.objective = objective
        self.regularizers = regularizers

        # Register regularizers' compute modules as submodules so device placement works
        # TODO: Convert RegularizerConfig to nn.Module? Doing so might allow better compilation
        for idx, reg in enumerate(self.regularizers):
            # Name is stable and readable in checkpoints
            self.add_module(f'reg_{reg.name}_{idx}', reg.compute_loss_term)
        self.timeline = Timeline(dopesheet)

        # Store the pure neural network model (passed as parameter)
        self.model = model

        # Set up latent capture hooks for all unique layer names
        self.latent_hooks: dict[str, ActivationCaptureHook] = {}
        self.hook_handles: list[RemovableHandle] = []

        # Names of labels that are currently "active" (used by any on-schedule regularizer)
        # Updated each training step; consumed by callbacks for conditional logging.
        self.active_labels: set[str] = set()

    def _setup_latent_hooks(self):
        """Set up hooks for all unique layers specified in regularizer layer_affinities."""
        for reg in self.regularizers:
            if len(reg.layer_affinities) == 0:
                log.warning(f'Regularizer {reg.name} has no layer affinities and will not run')
            if reg.name not in self.timeline.props and reg.train:
                log.warning(f'Regularizer {reg.name} is not in the timeline and will have full weight')

        regularized_layers = {
            name  #
            for reg in self.regularizers
            for name in reg.layer_affinities
        }

        for layer_name in regularized_layers:
            try:
                layer_module = self.model.get_submodule(layer_name)
            except AttributeError as e:
                reg_names = [reg.name for reg in self.regularizers if layer_name in reg.layer_affinities]
                raise AttributeError(f'Layer {layer_name} not found in model; needed by {", ".join(reg_names)}') from e
            hook = ActivationCaptureHook()
            handle = layer_module.register_forward_hook(hook)
            self.latent_hooks[layer_name] = hook
            self.hook_handles.append(handle)
            log.debug(f'Registered hook for layer: {layer_name}')

    @override
    def on_fit_start(self):
        """Called at the very beginning of fit. Set up hooks here for DDP compatibility."""
        super().on_fit_start()
        self._setup_latent_hooks()

    @override
    def on_fit_end(self):
        """Called at the very end of fit. Clean up hooks."""
        super().on_fit_end()
        for handle in self.hook_handles:
            handle.remove()
        self.latent_hooks.clear()
        self.hook_handles.clear()

    @override
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        log.info('Saving timeline state')
        checkpoint['timeline_step'] = self.timeline._step

    @override
    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
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

        # Calculate primary reconstruction loss
        primary_loss = self.objective(outputs, batch_data).mean()
        losses = {'recon': primary_loss.item()}
        total_loss = primary_loss

        # Determine which labels are "active" this step, i.e., referenced by at least one
        # regularizer that is training and has a non-zero weight in the current schedule.
        # Expose this for callbacks to optionally filter their logging.
        self.active_labels = {
            label
            for reg in self.regularizers
            if reg.train and self.schedule.get(reg.name, 1.0) != 0 and reg.label_affinities is not None
            for label in reg.label_affinities.keys()
        }

        # Calculate regularizer losses
        for reg in self.regularizers:
            # Respect training flag
            if not reg.train:
                continue
            # Default to full weight for regularizers that aren't in the timeline
            weight = self.schedule.get(reg.name, 1.0)
            if weight == 0:
                # Regularizer has been disabled for this timestep
                continue

            # Apply regularizer to each specified layer
            for layer_name in reg.layer_affinities:
                hook = self.latent_hooks[layer_name]
                if hook.activations is None:
                    log.warning(f'No latents captured for layer {layer_name}, skipping regularizer {reg.name}')
                    continue

                term_loss = compute_loss_term(reg, batch_labels, hook.activations)
                if term_loss is None:
                    continue

                if not torch.isfinite(term_loss):
                    raise RuntimeError(f'Loss term {reg.name}:{layer_name} at step {self.global_step} is {term_loss:f}')

                # Use layer-specific loss name for logging
                loss_key = f'{reg.name}:{layer_name}' if len(reg.layer_affinities) > 1 else reg.name
                losses[loss_key] = term_loss.item()
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
        super().on_before_optimizer_step(optimizer)
        current_lr = self.schedule.get('lr', 1e-8)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    @override
    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        if self.timeline.state.is_phase_start:
            self.print(f'Starting phase: {self.timeline.state.phase}')

    @override
    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.global_step < len(self.timeline) - 1:
            self.timeline.step()

    @override
    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        # Ensure hooks are available for capturing latents during validation.
        # If we're running validation inside fit, hooks are already set up in on_fit_start.
        # For standalone validate(), ensure they are registered.
        if not self.latent_hooks:
            self._setup_latent_hooks()

    @override
    def on_validation_end(self) -> None:
        """Called at the end of validation."""
        # Keep hooks intact if we're in a fit loop; they'll be removed in on_fit_end.
        # If we ever set them up specifically for standalone validation, we could
        # clean them here. For now, no action needed.

    @override
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Operates on a single batch of data from the validation set.

        In this step you'd might generate examples or calculate anything of interest like accuracy.
        """
        batch_data, batch_labels = batch
        assert all(isinstance(k, str) and isinstance(v, Tensor) for k, v in batch_labels.items())

        # Forward pass
        outputs: Tensor = self(batch_data)

        # Primary reconstruction/objective loss (reported as val_loss)
        primary_loss = self.objective(outputs, batch_data).mean()
        losses: dict[str, float] = {'recon': primary_loss.item()}

        # Compute regularizer metrics (do not add to val_loss); respect validate flag
        for reg in self.regularizers:
            if not reg.validate:
                continue

            for layer_name in reg.layer_affinities:
                hook = self.latent_hooks.get(layer_name)
                if hook is None or hook.activations is None:
                    log.warning(
                        f'No latents captured for layer {layer_name}, skipping validation metric for {reg.name}'
                    )
                    continue

                term_loss = compute_loss_term(reg, batch_labels, hook.activations)
                if term_loss is None:
                    continue

                if not torch.isfinite(term_loss):
                    raise RuntimeError(
                        f'Validation loss term {reg.name}:{layer_name} at step {self.global_step} is {term_loss:f}'
                    )

                loss_key = f'{reg.name}:{layer_name}' if len(reg.layer_affinities) > 1 else reg.name
                losses[loss_key] = term_loss.item()

        # Log metrics
        self.log('val_loss', primary_loss, prog_bar=True)
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value)

        return {'loss': primary_loss, 'losses': losses}


def compute_loss_term(regularizer: RegularizerConfig, batch_labels: dict[str, Tensor], latents: Tensor):
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
    # Align devices to avoid device-mixing errors
    if sample_affinities.device != per_sample_loss.device:
        sample_affinities = sample_affinities.to(per_sample_loss.device)
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
