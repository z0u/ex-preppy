import logging
from typing import Any

import torch
import torch.optim as optim
from ignite.engine import Engine, Events
from torch import Tensor
from torch.utils.data import DataLoader

from ex_color.criteria.criteria import LossCriterion, RegularizerConfig
from ex_color.events import StepMetricsEvent
from ex_color.loaders import reiterate
from ex_color.model import ColorMLP
from ex_color.result import InferenceResult
from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline

log = logging.getLogger(__name__)


def create_train_step(
    model: ColorMLP,
    optimizer: optim.Optimizer,
    loss_criterion: LossCriterion,
    regularizers: list[RegularizerConfig],
    timeline: Timeline,
    device: torch.device,
):
    """Create a training step function for Ignite Engine."""
    
    def train_step(engine: Engine, batch: tuple[Tensor, dict[str, Tensor]]) -> dict[str, Any]:
        model.train()
        batch_data, batch_labels = batch
        
        # Get current timeline state
        current_state = timeline.state
        
        # Update learning rate from dopesheet
        current_lr = current_state.props['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        
        # Forward pass - model now only returns outputs
        outputs = model(batch_data)
        
        # Get latents from hook
        latents = model.get_latents()
        if latents is None:
            raise RuntimeError("Latents not captured. Make sure hook is registered.")
        
        current_results = InferenceResult(outputs, latents)
        
        # Calculate primary reconstruction loss
        primary_loss = loss_criterion(batch_data, current_results).mean()
        losses = {'recon': primary_loss.item()}
        total_loss = primary_loss
        zeros = torch.tensor(0.0, device=batch_data.device)
        
        # Apply regularizers
        for regularizer in regularizers:
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
                # If the loss is a scalar, expand it to match the batch size
                per_sample_loss = per_sample_loss.expand(batch_data.shape[0])
            assert per_sample_loss.shape[0] == batch_data.shape[0], f'Loss should be per-sample OR scalar: {name}'
            
            # Apply sample affinities
            weighted_loss = per_sample_loss * sample_affinities
            
            # Calculate mean only over selected samples
            term_loss = weighted_loss.sum() / (sample_affinities.sum() + 1e-8)
            
            losses[name] = term_loss.item()
            if not torch.isfinite(term_loss):
                log.warning(f'Loss term {name} at step {engine.state.iteration} is not finite: {term_loss}')
                continue
            total_loss += term_loss * weight
        
        if total_loss > 0:
            total_loss.backward()
            optimizer.step()
        
        # Advance timeline after each iteration
        if engine.state.iteration < len(timeline):
            timeline.step()
        
        return {
            'total_loss': total_loss.item(),
            'losses': losses,
            'timeline_state': current_state,
            'batch_data': batch_data,
            'outputs': outputs,
            'latents': latents,
        }
    
    return train_step


def train_color_model_ignite(
    model: ColorMLP,
    train_loader: DataLoader,
    val_data: Tensor,
    dopesheet: Dopesheet,
    loss_criterion: LossCriterion,
    regularizers: list[RegularizerConfig],
    progress_bar=None,
    metrics_recorder=None,
):
    """Train the color model using PyTorch Ignite."""
    
    # Validate inputs
    if 'lr' not in dopesheet.props:
        raise ValueError("Dopesheet must define the 'lr' property column.")
    
    timeline = Timeline(dopesheet)
    optimizer = optim.Adam(model.parameters(), lr=0)
    device = next(model.parameters()).device
    
    # Register the latent hook
    model.register_latent_hook()
    
    try:
        # Create data iterator
        train_data = iter(reiterate(train_loader))
        
        # Create training step function
        train_step = create_train_step(
            model, optimizer, loss_criterion, regularizers, timeline, device
        )
        
        # Create Ignite engine
        trainer = Engine(train_step)
        
        # Add event handlers for progress tracking and metrics
        @trainer.on(Events.ITERATION_STARTED)
        def handle_iteration_started(engine: Engine):
            current_state = timeline.state
            
            # Handle phase start events
            if current_state.is_phase_start:
                if progress_bar is not None:
                    progress_bar.set_description(current_state.phase)
                log.debug(f'Phase started: {current_state.phase}')
            
            # Handle actions
            for action in current_state.actions:
                log.debug(f'Action: {action}')
        
        @trainer.on(Events.ITERATION_COMPLETED)
        def handle_iteration_completed(engine: Engine):
            if progress_bar is not None:
                progress_bar.update()
                
            # Update progress bar with current loss
            output = engine.state.output
            if progress_bar is not None and 'total_loss' in output:
                progress_bar.set_postfix({'train-loss': output['total_loss']})
            
            # Record metrics
            if metrics_recorder is not None:
                # Create a proper StepMetricsEvent for compatibility
                step_metrics_event = StepMetricsEvent(
                    name='step-metrics',
                    step=engine.state.iteration - 1,  # 0-indexed
                    model=model,
                    timeline_state=output['timeline_state'],
                    optimizer=optimizer,
                    train_batch=output['batch_data'],
                    total_loss=output['total_loss'],
                    losses=output['losses'],
                )
                metrics_recorder(step_metrics_event)
        
        @trainer.on(Events.ITERATION_COMPLETED)
        def handle_phase_end(engine: Engine):
            current_state = timeline.state
            
            # Handle phase end events
            if current_state.is_phase_end:
                log.debug(f'Phase ended: {current_state.phase}')
                
                # Perform validation
                with torch.no_grad():
                    model.eval()
                    val_outputs = model(val_data.to(device))
                    val_latents = model.get_latents()
                    
                    if val_latents is not None:
                        log.debug(f'Validation completed for phase: {current_state.phase}')
        
        # Calculate total iterations needed
        total_steps = len(timeline)
        
        # Create a custom data loader that yields from our iterator
        def custom_data_loader():
            for _ in range(total_steps):
                yield next(train_data)
        
        # Run training
        trainer.run(custom_data_loader(), max_epochs=1)
        
        log.debug('Training finished!')
        
    finally:
        # Always remove the hook
        model.remove_latent_hook()