"""
Modify activations at inference time to control model behavior.

Design notes:

- Interventions are nn.Module so they can be moved across devices with .to() and
    participate in state_dict()/pickling. Any Tensor state should be registered
    as buffers so it's saved and moved but not trained.
- Mappers are also nn.Module subclasses for consistent device handling and
  serialization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence, override

import torch.nn as nn
from torch import Tensor


@dataclass
class ConstAnnotation:
    direction: Literal['input', 'output']
    kind: Literal['linear', 'angular']
    name: str
    value: float


@dataclass
class VarAnnotation:
    name: str
    values: Tensor


class Mapper(nn.Module, ABC):
    """Protocol for mapper functions that modify alignment values."""

    @override  # Overridden to narrow types
    def __call__(self, alignment: Tensor) -> Tensor:
        return super().__call__(alignment)

    @override
    @abstractmethod
    def forward(self, alignment: Tensor) -> Tensor: ...

    @property
    def annotations(self) -> Sequence[ConstAnnotation]:
        """Annotations for visualization of hyperparameters"""
        ...


class Intervention(nn.Module, ABC):
    """Modify activations to suppress or amplify concepts."""

    kind: Literal['linear', 'rotational']

    def __init__(self):
        super().__init__()

    @abstractmethod
    def dist(self, activations: Tensor) -> Tensor:
        """Calculate the distance between the activations and the intervention region"""
        ...

    @override  # Overridden to narrow types
    def __call__(self, alignment: Tensor) -> Tensor:
        return super().__call__(alignment)

    @override
    @abstractmethod
    def forward(self, activations: Tensor, /) -> Tensor:
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
    def annotations(self) -> Sequence[ConstAnnotation]:
        """Annotations for visualization of hyperparameters"""
        ...

    @abstractmethod
    def annotate_activations(self, activations: Tensor) -> VarAnnotation:
        """Annotations for visualization of intervention effect"""
        ...


@dataclass
class InterventionConfig:
    """Configuration for an intervention."""

    apply: Intervention
    """Function to apply to the activations."""
    layer_affinities: list[str]
    """List of layer names to apply this intervention to."""
