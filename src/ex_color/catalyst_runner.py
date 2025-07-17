"""Standard Catalyst training using Runner."""

import logging

import torch
import torch.nn as nn
from catalyst.dl import Runner, Callback, CallbackOrder
from torch.utils.data import DataLoader

from ex_color.criteria.criteria import RegularizerConfig
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
            if hasattr(runner, 'optimizer') and runner.optimizer is not None:
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

    def compute_regularization_loss(self, batch_data, batch_labels, outputs, latents, regularizer_weights):
        """Compute regularization loss and return it as a tensor."""
        if latents is None:
            return None
        
        # Create inference result with captured latents
        current_results = InferenceResult(outputs, latents)
        
        total_reg_loss = 0.0
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
        
        # Store losses in batch_metrics for metrics recording
        if hasattr(self, '_runner_ref') and self._runner_ref is not None:
            if total_reg_loss > 0:
                self._runner_ref.batch_metrics['reg_loss'] = total_reg_loss.item()
                self._runner_ref.batch_metrics.update(reg_losses)
        
        return total_reg_loss if total_reg_loss > 0 else None


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


class ColorTransformerRunner(Runner):
    """Custom Runner for color transformer training."""

    def __init__(self, model, optimizer, criterion, latent_hook, max_steps, callbacks=None):
        super().__init__(model=model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.latent_hook = latent_hook
        self.max_steps = max_steps
        self.sample_step = 0
        self.callbacks = callbacks or []
        
        # Set reference to runner in regularizer callbacks
        for callback in self.callbacks:
            if isinstance(callback, RegularizerCallback):
                callback._runner_ref = self

    def handle_batch(self, batch):
        """Process a single batch."""
        # Store batch for callbacks
        self.batch = batch
        
        # Extract features
        features = batch['features']
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(features)
        
        # Store outputs in batch for callbacks
        self.batch['logits'] = outputs
        
        # Calculate main loss (MSE between input and output)
        main_loss = self.criterion(outputs, features)
        
        # Store main loss
        self.batch_metrics = {'loss': main_loss.item()}
        
        # Initialize total loss tensor for backward pass
        total_loss = main_loss
        
        # Add regularization through regularizer callback
        # We need to simulate the callback execution here for proper integration
        for callback in self.callbacks:
            if isinstance(callback, RegularizerCallback):
                # Get regularization losses
                reg_loss = callback.compute_regularization_loss(self.batch['features'], 
                                                              self.batch.get('targets', {}),
                                                              outputs,
                                                              self.latent_hook.latents,
                                                              getattr(self, 'regularizer_weights', {}))
                if reg_loss is not None:
                    total_loss = total_loss + reg_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Update step counter
        self.sample_step += 1
        
        # Stop training after the correct number of steps
        if hasattr(self, 'max_steps') and self.sample_step >= self.max_steps:
            self.need_early_stop = True


def train_color_model_catalyst_runner(
    model: ColorMLP,
    train_loader: DataLoader,
    val_data: torch.Tensor,
    dopesheet: Dopesheet,
    regularizers: list[RegularizerConfig],
    metrics_recorder: MetricsRecorder | None = None,
):
    """Train the color model using standard Catalyst Runner."""
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

        # Create MSE loss criterion (simplified as suggested)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initial LR, will be overridden

        # Calculate total number of steps
        total_steps = len(Timeline(dopesheet))

        # Create custom runner
        runner = ColorTransformerRunner(model, optimizer, criterion, latent_hook, total_steps, callbacks)

        # Run training
        runner.train(
            loaders={'train': train_loader},
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=1,  # We'll control iterations via callbacks
            callbacks=callbacks,
            verbose=False,  # Disable verbose to avoid issues
            logdir=None,  # No logging to disk
        )

    finally:
        # Clean up hook
        hook_handle.remove()

    return model