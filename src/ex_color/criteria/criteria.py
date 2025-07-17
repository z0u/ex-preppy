import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from ex_color.result import InferenceResult

log = logging.getLogger(__name__)


@runtime_checkable
class LossCriterion(Protocol):
    def __call__(self, data: Tensor, res: InferenceResult) -> Tensor: ...


@dataclass
class RegularizerConfig:
    """Configuration for a regularizer, including label affinities."""

    name: str
    """Matched with hyperparameter for weighting"""
    criterion: LossCriterion
    label_affinities: dict[str, float] | None
    """Maps label names to affinity strengths"""
