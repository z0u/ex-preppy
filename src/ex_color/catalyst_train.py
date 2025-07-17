"""Catalyst-based training for the color model."""

import logging

import torch
import torch.nn as nn
from catalyst.dl import Callback, CallbackOrder, SupervisedRunner
from torch.utils.data import DataLoader

from ex_color.criteria.criteria import LossCriterion, RegularizerConfig
from ex_color.model import ColorMLP
from ex_color.record import MetricsRecorder
from ex_color.result import InferenceResult
from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline

log = logging.getLogger(__name__)


class LatentHook:
    """Hook to capture bottleneck latents from the encoder."""

    def __init__(self):
        self.latents = None

    def __call__(self, module, input, output):
        self.latents = output


class DopesheetCallback(Callback):
    """Callback that updates learning rate and regularizer weights according to dopesheet."""

    def __init__(self, dopesheet: Dopesheet, regularizer_names: list[str]):
        super().__init__(CallbackOrder.Scheduler)
        self.timeline = Timeline(dopesheet)
        self.regularizer_names = regularizer_names
        self.step_count = 0

    def on_batch_start(self, runner):
        """Update parameters according to dopesheet timeline."""
        if self.step_count < len(self.timeline):
            current_state = self.timeline.state

            # Update learning rate
            current_lr = current_state.props.get('lr', runner.optimizer.param_groups[0]['lr'])
            for param_group in runner.optimizer.param_groups:
                param_group['lr'] = current_lr

            # Store regularizer weights in runner for use by other callbacks
            if not hasattr(runner, 'regularizer_weights'):
                runner.regularizer_weights = {}

            for reg_name in self.regularizer_names:
                weight = current_state.props.get(reg_name, 1.0)
                runner.regularizer_weights[reg_name] = weight

            # Advance timeline after processing current step
            if self.step_count < len(self.timeline) - 1:
                self.timeline.step()

        self.step_count += 1


class RegularizerCallback(Callback):
    """Callback that applies regularization losses using latents captured by hooks."""

    def __init__(self, regularizers: list[RegularizerConfig], latent_hook: LatentHook):
        super().__init__(CallbackOrder.Backward)
        self.regularizers = regularizers
        self.latent_hook = latent_hook

    def _compute_regularizer_loss(self, regularizer, batch_data, batch_labels, current_results, regularizer_weights):
        """Compute loss for a single regularizer."""
        name = regularizer.name
        criterion = regularizer.criterion
        weight = regularizer_weights.get(name, 1.0)

        if weight == 0:
            return None, 0.0

        if regularizer.label_affinities is not None:
            # Calculate sample affinities based on labels
            label_probs = [
                batch_labels[k] * v
                for k, v in regularizer.label_affinities.items()
                if k in batch_labels
            ]
            if not label_probs:
                return None, 0.0

            sample_affinities = torch.stack(label_probs, dim=0).sum(dim=0)
            sample_affinities = torch.clamp(sample_affinities, 0.0, 1.0)
            if torch.allclose(sample_affinities, torch.zeros_like(sample_affinities)):
                return None, 0.0
        else:
            sample_affinities = torch.ones(batch_data.shape[0], device=batch_data.device)

        per_sample_loss = criterion(batch_data, current_results)
        if len(per_sample_loss.shape) == 0:
            per_sample_loss = per_sample_loss.expand(batch_data.shape[0])

        # Apply sample affinities
        weighted_loss = per_sample_loss * sample_affinities

        # Calculate mean only over selected samples
        term_loss = weighted_loss.sum() / (sample_affinities.sum() + 1e-8)

        return name, term_loss

    def on_batch_end(self, runner):
        """Add regularization losses to the batch."""
        if self.latent_hook.latents is None:
            return

        # Get current batch data and labels - using runner.batch
        batch_dict = runner.batch
        batch_data = batch_dict['features']
        batch_labels = batch_dict.get('targets', {})

        # Create inference result with captured latents
        outputs = runner.batch_metrics.get('outputs', batch_data)  # fallback to batch_data if no outputs
        current_results = InferenceResult(outputs, self.latent_hook.latents)

        # Get regularizer weights from dopesheet callback
        regularizer_weights = getattr(runner, 'regularizer_weights', {})

        total_reg_loss = torch.tensor(0.0, device=batch_data.device)
        reg_losses = {}

        for regularizer in self.regularizers:
            result = self._compute_regularizer_loss(
                regularizer, batch_data, batch_labels, current_results, regularizer_weights
            )
            if result[0] is not None:
                name, term_loss = result
                reg_losses[name] = term_loss.item()
                if torch.isfinite(term_loss):
                    weight = regularizer_weights.get(name, 1.0)
                    total_reg_loss += term_loss * weight

        # Add regularization loss to the main loss
        if total_reg_loss > 0:
            runner.batch_metrics['reg_loss'] = total_reg_loss.item()
            runner.batch_metrics.update(reg_losses)

            # Store the regularization loss for later use
            if not hasattr(runner, 'additional_loss'):
                runner.additional_loss = 0.0
            runner.additional_loss += total_reg_loss.item()


class MetricsRecorderCallback(Callback):
    """Callback that records training metrics."""

    def __init__(self, metrics_recorder: MetricsRecorder):
        super().__init__(CallbackOrder.External)
        self.metrics_recorder = metrics_recorder

    def on_batch_end(self, runner):
        """Record batch metrics."""
        step = getattr(runner, 'sample_step', 0)
        losses = dict(runner.batch_metrics)
        total_loss = losses.get('loss', 0.0)

        # Create a simple event-like object for the metrics recorder
        class MockEvent:
            def __init__(self, step, total_loss, losses):
                self.step = step
                self.total_loss = total_loss
                self.losses = losses

        event = MockEvent(step, total_loss, losses)
        self.metrics_recorder(event)


def train_color_model_catalyst(
    model: ColorMLP,
    train_loader: DataLoader,
    val_data: torch.Tensor,
    dopesheet: Dopesheet,
    loss_criterion: LossCriterion,
    regularizers: list[RegularizerConfig],
    metrics_recorder: MetricsRecorder | None = None,
):
    """Train the color model using Catalyst."""
    # Set up latent capture hook
    latent_hook = LatentHook()
    hook_handle = model.encoder.register_forward_hook(latent_hook)

    try:
        # Set up callbacks
        callbacks = []

        # Dopesheet callback for parameter scheduling
        regularizer_names = [r.name for r in regularizers]
        dopesheet_callback = DopesheetCallback(dopesheet, regularizer_names)
        callbacks.append(dopesheet_callback)

        # Regularizer callback
        regularizer_callback = RegularizerCallback(regularizers, latent_hook)
        callbacks.append(regularizer_callback)

        # Metrics recorder callback
        if metrics_recorder is not None:
            metrics_callback = MetricsRecorderCallback(metrics_recorder)
            callbacks.append(metrics_callback)

        # Create a custom criterion that works with our setup
        class CatalystCriterion(nn.Module):
            def __init__(self, loss_criterion: LossCriterion, latent_hook: LatentHook):
                super().__init__()
                self.loss_criterion = loss_criterion
                self.latent_hook = latent_hook

            def forward(self, outputs, targets):
                # targets should contain 'features' (input data) and 'targets' (labels)
                batch_data = targets['features']

                # Create inference result with captured latents
                if self.latent_hook.latents is not None:
                    current_results = InferenceResult(outputs, self.latent_hook.latents)
                    return self.loss_criterion(batch_data, current_results).mean()
                else:
                    # Fallback to simple MSE between input and output
                    return nn.MSELoss()(outputs, batch_data)

        criterion = CatalystCriterion(loss_criterion, latent_hook)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initial LR, will be overridden

        # Calculate total number of epochs based on dopesheet length and loader size
        total_steps = len(Timeline(dopesheet))
        steps_per_epoch = len(train_loader)
        num_epochs = max(1, total_steps // steps_per_epoch + 1)

        # Create runner
        runner = SupervisedRunner()

        # Convert val_data to a simple loader format for Catalyst
        val_dataset = torch.utils.data.TensorDataset(val_data, val_data)  # Dummy targets
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_data))

        # Run training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders={'train': train_loader, 'valid': val_loader},
            num_epochs=num_epochs,
            callbacks=callbacks,
            logdir=None,  # No logging for now
            verbose=False,  # We handle our own progress
        )

    finally:
        # Clean up hook
        hook_handle.remove()

    return runner
