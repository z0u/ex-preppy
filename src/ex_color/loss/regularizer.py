import logging
from dataclasses import dataclass
from typing import Literal

from torch import Tensor
import torch.nn as nn


log = logging.getLogger(__name__)


class Regularizer(nn.Module):
    """
    Compute a loss term based on hidden layer activations.

    Args:
        activations (Tensor): The activations to regularize, typically of shape [B, ...].

    Returns:
        Tensor: A tensor of shape [B] containing the regularization loss for each sample, or [] (scalar) for mean loss.
    """

    def __call__(self, activations: Tensor, /) -> Tensor: ...


Phase = Literal['train', 'validate']


@dataclass
class RegularizerConfig:
    """Configuration for a regularizer, including label and layer affinities."""

    name: str
    """Matched with hyperparameter for weighting"""
    compute_loss_term: Regularizer
    """Function to compute a loss term based on hidden layer activations"""
    label_affinities: dict[str, float] | None
    """Maps label names to affinity strengths"""
    layer_affinities: list[str]
    """List of layer names to apply this regularizer to, e.g. ['encoder', 'decoder.0']"""
    phase: Phase | tuple[Phase, ...] = 'train'

    @property
    def train(self):
        """Regularizer is active in training (fit) phase"""
        return 'train' == self.phase or 'train' in self.phase

    @property
    def validate(self):
        """Emits loss as metric in validation phase"""
        return 'validate' == self.phase or 'validate' in self.phase
