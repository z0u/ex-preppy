import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


log = logging.getLogger(__name__)


@runtime_checkable
class Regularizer(Protocol):
    """
    Compute a loss term based on hidden layer activations.

    Args:
        activations (Tensor): The activations to regularize, typically of shape [B, ...].

    Returns:
        Tensor: A tensor of shape [B] containing the regularization loss for each sample, or [] (scalar) for mean loss.
    """

    def __call__(self, activations: Tensor, /) -> Tensor: ...


@dataclass
class RegularizerConfig:
    """Configuration for a regularizer, including label affinities."""

    name: str
    """Matched with hyperparameter for weighting"""
    compute_loss_term: Regularizer
    """Function to compute a loss term based on hidden layer activations"""
    label_affinities: dict[str, float] | None
    """Maps label names to affinity strengths"""
