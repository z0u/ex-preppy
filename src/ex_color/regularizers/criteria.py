import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


log = logging.getLogger(__name__)


@runtime_checkable
class Regularizer(Protocol):
    def __call__(self, activations: Tensor) -> Tensor: ...


@dataclass
class RegularizerConfig:
    """Configuration for a regularizer, including label affinities."""

    name: str
    """Matched with hyperparameter for weighting"""
    criterion: Regularizer
    label_affinities: dict[str, float] | None
    """Maps label names to affinity strengths"""
