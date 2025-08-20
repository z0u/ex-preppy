"""Interventions modify activations at inference time to control model behavior."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

from torch import Tensor


@dataclass
class ConstAnnotation:
    direction: Literal['input', 'output']
    type: Literal['linear', 'angular']
    name: str
    value: float


@dataclass
class VarAnnotation:
    name: str
    values: Tensor


class Intervention(ABC):
    """Modify activations to suppress or amplify concepts."""

    type: Literal['linear', 'rotational', 'other']

    def __init__(self, concept_vector: Tensor):
        self.concept_vector = concept_vector

    @abstractmethod
    def dist(self, activations: Tensor) -> Tensor: ...

    @abstractmethod
    def __call__(self, activations: Tensor, /) -> Tensor:
        """
        Modify activations to suppress or amplify concepts.

        Args:
            activations: The activations to modify, typically of shape [B, ...].

        Returns:
            Tensor: The modified activations, with the same shape as the input.
        """
        ...

    @property
    @abstractmethod
    def annotations(self) -> Sequence[ConstAnnotation]: ...

    @abstractmethod
    def annotate_activations(self, activations: Tensor) -> VarAnnotation: ...


@dataclass
class InterventionConfig:
    """Configuration for an intervention."""

    name: str
    """Name for this intervention, for reference."""
    apply: Intervention
    """Function to apply to the activations."""
    layer_affinities: list[str]
    """List of layer names to apply this intervention to."""
    strength: float = 1.0
    """Strength of the intervention, from 0.0 (no effect) to 1.0 (full effect)."""
