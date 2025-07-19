from dataclasses import dataclass


@dataclass(slots=True)
class DynamicPropState:
    value: float
    velocity: float
    acceleration: float


@dataclass(slots=True)
class Frame:
    step: int
    phase: str
    actions: list[str]
    props: dict[str, float]
    is_phase_start: bool
    is_phase_end: bool
