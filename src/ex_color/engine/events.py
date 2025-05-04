from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    import torch
    import torch.optim as optim

    from mini.temporal.timeline import State

    from .types import InferenceResult


T = TypeVar('T')


@dataclass(eq=False, frozen=True)
class Event:
    name: str
    step: int
    model: 'torch.nn.Module'  # Use generic nn.Module
    timeline_state: 'State'
    optimizer: 'optim.Optimizer'


@dataclass(eq=False, frozen=True)
class PhaseEndEvent(Event):
    validation_data: 'torch.Tensor'
    inference_result: 'InferenceResult'


@dataclass(eq=False, frozen=True)
class StepMetricsEvent(Event):
    """Event carrying metrics calculated during a training step."""

    total_loss: float
    losses: dict[str, float]


class EventHandler[T](Protocol):
    def __call__(self, event: T) -> None: ...


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
