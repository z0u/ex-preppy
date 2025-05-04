import logging
from typing import Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mini.temporal.dopesheet import Dopesheet
from mini.temporal.timeline import Timeline
from utils.progress import RichProgress

from .events import Event, EventHandlers, PhaseEndEvent, StepMetricsEvent
from .types import InferenceResult, LossCriterion, SpecialLossCriterion

log = logging.getLogger(__name__)


def reiterate[T](it: Iterable[T]) -> Iterator[T]:
    """
    Iterates over an iterable indefinitely.

    When the iterable is exhausted, it starts over from the beginning. Unlike
    `itertools.cycle`, yielded values are not cached — so each iteration may be
    different.
    """
    while True:
        yield from it


def train_model(  # noqa: C901
    model: nn.Module,
    datasets: dict[str, tuple[DataLoader, torch.Tensor]],  # (train_loader, validation_tensor)
    dopesheet: Dopesheet,
    loss_criteria: dict[str, LossCriterion | SpecialLossCriterion],
    event_handlers: EventHandlers | None = None,
):
    """
    Generic training loop driven by a Dopesheet timeline and event handlers.

    Args:
        model: The neural network model (must return a tuple where the second element is latents if using Anchor regularizer).
        datasets: A dictionary mapping phase names to tuples of (training DataLoader, validation Tensor).
        dopesheet: The Dopesheet defining the curriculum schedule.
        loss_criteria: A dictionary mapping loss/regularizer names (matching dopesheet props) to their criterion functions.
        event_handlers: Optional EventHandlers instance for custom callbacks.
    """
    if event_handlers is None:
        event_handlers = EventHandlers()

    # --- Validate inputs ---
    dopesheet_phases = dopesheet.phases
    missing_data = dopesheet_phases - set(datasets.keys())
    if missing_data:
        raise ValueError(f'Missing data for dopesheet phases: {missing_data}')

    if 'lr' not in dopesheet.props:
        raise ValueError("Dopesheet must define the 'lr' property column.")
    # --- End Validation ---

    timeline = Timeline(dopesheet)
    # Initialize optimizer with lr=0; actual LR is set each step from timeline
    optimizer = optim.Adam(model.parameters(), lr=0)
    device = next(model.parameters()).device
    model.to(device)

    data_iterators = {phase_name: iter(reiterate(dataloader)) for phase_name, (dataloader, _) in datasets.items()}

    total_steps = len(timeline)

    with RichProgress(total=total_steps, description='Training Steps') as pbar:
        for step in range(total_steps):
            # Get state *before* advancing timeline for this step's processing
            current_state = timeline.state
            current_phase_name = current_state.phase

            # Get training batch for the current phase
            try:
                # Assuming TensorDataset yields a tuple with one element (the data tensor)
                (batch,) = next(data_iterators[current_phase_name])
                batch = batch.to(device)
            except StopIteration:
                log.error(f"DataLoader for phase '{current_phase_name}' exhausted unexpectedly.")
                # Re-initialize iterator (shouldn't happen with reiterate, but defensive)
                data_iterators[current_phase_name] = iter(reiterate(datasets[current_phase_name][0]))
                (batch,) = next(data_iterators[current_phase_name])
                batch = batch.to(device)

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
            model.train()  # Ensure model is in training mode

            current_lr = current_state.props['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # --- Forward pass ---
            # We need to handle different model output signatures.
            # Let's assume the model returns (outputs, latents) for compatibility
            # with Anchor and other potential latent-based losses.
            # Users with different models might need to wrap them or adjust losses.
            model_output = model(batch)
            if not (isinstance(model_output, tuple) and len(model_output) >= 2):
                log.warning(
                    f'Model output is not a tuple of length >= 2. Assuming output is the first element and latents are the second. Got: {type(model_output)}'
                )
                # Attempt to handle common cases, but this is fragile.
                if isinstance(model_output, torch.Tensor):
                    outputs = model_output
                    # Cannot get latents if model only returns outputs
                    # This will break regularizers that need latents (like Anchor, unitary, planarity)
                    # We should probably raise an error or make the latent requirement explicit.
                    # For now, let's create a dummy tensor, but this needs refinement.
                    latents = torch.empty(0, device=device)  # Placeholder
                    log.error('Model only returned one tensor. Latent-based regularizers will likely fail.')
                else:
                    raise TypeError(
                        f'Unsupported model output type: {type(model_output)}. Expected tuple (outputs, latents, ...)'
                    )
            else:
                outputs, latents = model_output[0], model_output[1]

            current_results = InferenceResult(outputs, latents)
            # --- End Forward pass ---

            # --- Loss Calculation ---
            total_loss = torch.tensor(0.0, device=device)
            losses_dict: dict[str, float] = {}
            for name, criterion in loss_criteria.items():
                weight = current_state.props.get(name, 0.0)
                if weight == 0:
                    continue

                term_loss: torch.Tensor | None = None
                if isinstance(criterion, SpecialLossCriterion):
                    # Special criteria might run on their own data (like Anchor)
                    # The forward method gets the model and the *current batch*
                    # It returns an InferenceResult based on its *internal* data (e.g., anchor points)
                    special_results = criterion.forward(model, batch)
                    if special_results is not None:
                        # The __call__ method compares the special_results (e.g., current latents for anchor points)
                        # to the target (e.g., stored anchor latents).
                        # The 'batch' argument to __call__ is often ignored by SpecialLossCriterion implementations.
                        term_loss = criterion(batch, special_results)
                elif isinstance(criterion, LossCriterion):
                    # Standard criteria operate on the results of the current batch
                    term_loss = criterion(batch, current_results)
                else:
                    log.warning(f"Item '{name}' in loss_criteria is not a valid LossCriterion.")
                    continue

                if term_loss is not None and torch.isfinite(term_loss):
                    total_loss += term_loss * weight
                    losses_dict[name] = term_loss.item()
                elif term_loss is not None:
                    log.warning(f"Non-finite loss term '{name}': {term_loss.item()}. Skipping.")
                    losses_dict[name] = term_loss.item()  # Record it anyway for debugging

            # --- End Loss Calculation ---

            # --- Backward Pass & Optimizer Step ---
            if total_loss > 0 and torch.isfinite(total_loss):
                total_loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            elif not torch.isfinite(total_loss):
                log.error(f'Total loss is not finite ({total_loss.item()}) at step {step}. Skipping optimizer step.')
                # Optionally zero grad anyway to prevent accumulation from previous steps?
                optimizer.zero_grad()

            # --- End Training Step ---

            # Emit step metrics event
            step_metrics_event = StepMetricsEvent(
                name='step-metrics',
                **event_template,
                total_loss=total_loss.item() if torch.isfinite(total_loss) else float('nan'),
                losses=losses_dict,
            )
            event_handlers.step_metrics.emit('step-metrics', step_metrics_event)

            # --- Post-Step Event Handling & Validation ---
            if current_state.is_phase_end:
                model.eval()
                # Trigger phase-end for the *current* phase
                _, validation_data = datasets[current_phase_name]
                validation_data = validation_data.to(device)
                with torch.no_grad():
                    # Assuming model returns (outputs, latents) tuple
                    val_output = model(validation_data)
                    if not (isinstance(val_output, tuple) and len(val_output) >= 2):
                        # Handle potential mismatch in validation as well
                        log.warning(
                            f'Validation: Model output is not a tuple of length >= 2. Assuming output is the first element and latents are the second. Got: {type(val_output)}'
                        )
                        if isinstance(val_output, torch.Tensor):
                            val_outputs = val_output
                            val_latents = torch.empty(0, device=device)  # Placeholder
                        else:
                            # Fallback or raise error
                            val_outputs = torch.empty(0, device=device)
                            val_latents = torch.empty(0, device=device)
                            log.error('Validation: Cannot determine outputs/latents from model output.')
                    else:
                        val_outputs, val_latents = val_output[0], val_output[1]

                event = PhaseEndEvent(
                    name=f'phase-end:{current_phase_name}',
                    **event_template,  # Note: model is still in event_template
                    validation_data=validation_data.cpu(),
                    inference_result=InferenceResult(val_outputs, val_latents).cpu(),
                )
                event_handlers.phase_end.emit(event.name, event)
                event_handlers.phase_end.emit('phase-end', event)
            # --- End Event Handling ---

            # Update progress bar
            pbar.update(
                metrics={
                    'PHASE': current_phase_name,
                    'lr': f'{current_lr:.6f}',
                    'loss': f'{total_loss.item():.4f}' if torch.isfinite(total_loss) else 'NaN',
                    **{
                        name: f'{lt:.4f}' if torch.isfinite(torch.tensor(lt)) else 'NaN'
                        for name, lt in losses_dict.items()
                    },
                },
            )

            # Advance timeline *after* processing the current step
            # Check step < total_steps - 1 because timeline length is number of steps,
            # and steps are 0-indexed. We process step `total_steps - 1` last.
            if step < total_steps - 1:
                timeline.step()
            elif step == total_steps - 1:
                log.info('Last timeline step processed.')

    log.info('Training finished!')
