import logging

from torch import Tensor

from ex_color.events import Event, EventHandler, StepMetricsEvent

log = logging.getLogger(__name__)


class ModelRecorder(EventHandler):
    """Event handler to record model parameters."""

    history: list[tuple[int, dict[str, Tensor]]]
    """A list of tuples (step, state_dict) where state_dict is a copy of the model's state dict."""

    def __init__(self):
        self.history = []

    def __call__(self, event: Event):
        # Get a *copy* of the state dict and move it to the CPU
        # so we don't hold onto GPU memory or track gradients unnecessarily.
        model_state = {k: v.cpu().clone() for k, v in event.model.state_dict().items()}
        self.history.append((event.step, model_state))
        log.debug(f'Recorded model state at step {event.step}')


class BatchRecorder(EventHandler):
    """Event handler to record the exact batches used at each step."""

    history: list[tuple[int, Tensor]]
    """A list of tuples (step, train_batch)."""

    def __init__(self):
        self.history = []

    def __call__(self, event: StepMetricsEvent):
        if not isinstance(event, StepMetricsEvent):
            log.warning(f'BatchRecorder received unexpected event type: {type(event)}')
            return
        self.history.append((event.step, event.train_batch.cpu().clone()))
        log.debug(f'Recorded batch at step {event.step}')


class MetricsRecorder(EventHandler):
    """Event handler to record training metrics."""

    history: list[tuple[int, float, dict[str, float]]]
    """A list of tuples (step, total_loss, losses_dict)."""

    def __init__(self):
        self.history = []

    def __call__(self, event: StepMetricsEvent):
        if not isinstance(event, StepMetricsEvent):
            log.warning(f'MetricsRecorder received unexpected event type: {type(event)}')
            return

        self.history.append((event.step, event.total_loss, event.losses.copy()))
        log.debug(f'Recorded metrics at step {event.step}: loss={event.total_loss:.4f}')
