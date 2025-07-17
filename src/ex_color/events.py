import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Protocol

import torch.optim as optim
from torch import Tensor

from ex_color.model import ColorMLP
from ex_color.result import InferenceResult
from mini.temporal.timeline import State

log = logging.getLogger(__name__)


@dataclass(eq=False, frozen=True)
class Event:
    name: str
    step: int
    model: ColorMLP
    timeline_state: State
    optimizer: optim.Optimizer


@dataclass(eq=False, frozen=True)
class PhaseEndEvent(Event):
    validation_data: Tensor
    inference_result: InferenceResult


@dataclass(eq=False, frozen=True)
class StepMetricsEvent(Event):
    """Event carrying metrics calculated during a training step."""

    train_batch: Tensor
    total_loss: float
    losses: dict[str, float]


class EventHandler[T](Protocol):
    def __call__(self, event: T) -> Any: ...


class EventBinding[T]:
    """A class to bind events to handlers."""

    def __init__(self, event_name: str):
        self.event_name = event_name
        self.handlers: list[tuple[str, EventHandler[T]]] = []

    def add_handler(self, event_name: str, handler: EventHandler[T]) -> None:
        self.handlers.append((event_name, handler))

    def emit(self, event_name: str, event: T) -> None:
        for name, handler in self.handlers:
            if name == event_name:
                handler(event)


class EventHandlers:
    """A simple event system to allow for custom callbacks."""

    phase_start: EventBinding[Event]
    pre_step: EventBinding[Event]
    action: EventBinding[Event]
    phase_end: EventBinding[PhaseEndEvent]
    step_metrics: EventBinding[StepMetricsEvent]

    def __init__(self):
        self.phase_start = EventBinding[Event]('phase-start')
        self.pre_step = EventBinding[Event]('pre-step')
        self.action = EventBinding[Event]('action')
        self.phase_end = EventBinding[PhaseEndEvent]('phase-end')
        self.step_metrics = EventBinding[StepMetricsEvent]('step-metrics')


def periodic[T: Event](
    handler: EventHandler[T], *, interval: int, offset: int = 0, use_step: bool = True
) -> EventHandler[T]:
    """Decorator to run a handler at regular intervals."""
    i = 0

    @wraps(handler)
    def handler_wrapper(event: T):
        nonlocal i
        if use_step:
            i = event.step
        try:
            if (i + offset) % interval == 0:
                handler(event)
        finally:
            i += 1

    return handler_wrapper
