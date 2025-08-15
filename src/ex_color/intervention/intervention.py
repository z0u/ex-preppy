"""Interventions modify activations at inference time to control model behavior."""

from dataclasses import dataclass

from torch import Tensor

from ex_color.intervention.falloff import Falloff


class Intervention:
    """Modify activations to suppress or amplify concepts."""

    def __init__(self, subject: Tensor, falloff: Falloff):
        self.subject = subject
        self.falloff = falloff

    def dist(self, activations: Tensor) -> Tensor: ...

    def __call__(self, activations: Tensor, /) -> Tensor:
        """
        Modify activations to suppress or amplify concepts.

        Args:
            activations: The activations to modify, typically of shape [B, ...].

        Returns:
            Tensor: The modified activations, with the same shape as the input.
        """
        ...


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
