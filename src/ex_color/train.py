import logging

import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from ex_color.criteria.criteria import LossCriterion, RegularizerConfig
from ex_color.events import Event, EventHandlers, PhaseEndEvent, StepMetricsEvent
from ex_color.loaders import reiterate
from ex_color.model import ColorMLP
from ex_color.result import InferenceResult
from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline

log = logging.getLogger(__name__)


def train_color_model(  # noqa: C901
    model: ColorMLP,
    train_loader: DataLoader,
    val_data: Tensor,
    dopesheet: Dopesheet,
    loss_criterion: LossCriterion,
    regularizers: list[RegularizerConfig],
    event_handlers: EventHandlers | None = None,
):
    if event_handlers is None:
        event_handlers = EventHandlers()

    # --- Validate inputs ---
    if 'lr' not in dopesheet.props:
        raise ValueError("Dopesheet must define the 'lr' property column.")
    # --- End Validation ---

    timeline = Timeline(dopesheet)
    optimizer = optim.Adam(model.parameters(), lr=0)
    device = next(model.parameters()).device

    train_data = iter(reiterate(train_loader))

    total_steps = len(timeline)

    for step in range(total_steps):
        # Get state *before* advancing timeline for this step's processing
        current_state = timeline.state
        current_phase_name = current_state.phase

        batch_data, batch_labels = next(train_data)
        # Should already be on device
        # batch_data = batch_data.to(device)
        # batch_labels = batch_labels.to(device)

        # --- Event Handling ---
        event_template = {
            'step': step,
            'model': model,
            'timeline_state': current_state,
            'optimizer': optimizer,
        }

        if current_state.is_phase_start:
            event = Event(name=f'phase-start:{current_phase_name}', **event_template)
            event_handlers.phase_start.emit(event.name, event)
            event_handlers.phase_start.emit('phase-start', event)

        for action in current_state.actions:
            event = Event(name=f'action:{action}', **event_template)
            event_handlers.action.emit(event.name, event)
            event_handlers.action.emit('action', event)

        event = Event(name='pre-step', **event_template)
        event_handlers.pre_step.emit('pre-step', event)

        # --- Training Step ---
        # ... (get data, update LR, zero grad, forward pass, calculate loss, backward, step) ...

        current_lr = current_state.props['lr']
        # REF_BATCH_SIZE = 32
        # lr_scale_factor = batch.shape[0] / REF_BATCH_SIZE
        # current_lr = current_lr * lr_scale_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()

        outputs, latents = model(batch_data)
        current_results = InferenceResult(outputs, latents)

        primary_loss = loss_criterion(batch_data, current_results).mean()
        losses = {'recon': primary_loss.item()}
        total_loss = primary_loss
        zeros = torch.tensor(0.0, device=batch_data.device)

        for regularizer in regularizers:
            name = regularizer.name
            criterion = regularizer.criterion

            weight = current_state.props.get(name, 1.0)
            if weight == 0:
                continue

            if regularizer.label_affinities is not None:
                # Soft labels that indicate how much effect this regularizer has, based on its affinity with the label
                label_probs = [
                    batch_labels[k] * v
                    for k, v in regularizer.label_affinities.items()
                    if k in batch_labels  #
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

            # Apply sample importance weights
            # weighted_loss *= batch_weights

            # Calculate mean only over selected samples. If we used torch.mean, it would average over all samples, including those with 0 weight
            term_loss = weighted_loss.sum() / (sample_affinities.sum() + 1e-8)

            losses[name] = term_loss.item()
            if not torch.isfinite(term_loss):
                log.warning(f'Loss term {name} at step {step} is not finite: {term_loss}')
                continue
            total_loss += term_loss * weight

        if total_loss > 0:
            total_loss.backward()
            optimizer.step()
        # --- End Training Step ---

        # Emit step metrics event
        step_metrics_event = StepMetricsEvent(
            name='step-metrics',
            **event_template,
            train_batch=batch_data,
            total_loss=total_loss.item(),
            losses=losses,
        )
        event_handlers.step_metrics.emit('step-metrics', step_metrics_event)

        # --- Post-Step Event Handling ---
        if current_state.is_phase_end:
            # Trigger phase-end for the *current* phase
            # validation_data = batch_data
            with torch.no_grad():
                val_outputs, val_latents = model(val_data.to(device))
            event = PhaseEndEvent(
                name=f'phase-end:{current_phase_name}',
                **event_template,
                validation_data=val_data,
                inference_result=InferenceResult(val_outputs, val_latents),
            )
            event_handlers.phase_end.emit(event.name, event)
            event_handlers.phase_end.emit('phase-end', event)
        # --- End Event Handling ---

        # Advance timeline *after* processing the current step
        if step < total_steps:  # Avoid stepping past the end
            timeline.step()

    log.debug('Training finished!')
