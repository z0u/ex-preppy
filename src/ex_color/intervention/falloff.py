from typing import override

import torch
from torch import Tensor


class Falloff:
    """
    Adjust intervention amount based on distance to subject.

    Args:
        dots: Cosine distances of the activations to the subject, of shape [B].

    Returns:
        The target cosine distances, of shape [B].
    """

    def __call__(self, dots: Tensor, /) -> Tensor: ...


class Linear(Falloff):
    """Linear decay: target_dot = original_dot * decay_factor"""

    def __init__(self, decay: float = 0.5):
        assert 0 <= decay <= 1, 'decay_factor must be in [0, 1]'
        self.decay = decay

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        return dots * self.decay

    def __repr__(self):
        return f'{type(self).__name__}(decay={self.decay})'


class Power(Falloff):
    """Produces a specular-like lobe."""

    def __init__(self, power: float = 10):
        self.power = power

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        return dots**self.power

    def __repr__(self):
        return f'{type(self).__name__}({self.power})'


class Sinus(Falloff):
    """Produces a sinusoidal lobe."""

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        return torch.sin(dots * torch.pi / 2)

    def __repr__(self):
        return f'{type(self).__name__}()'


class Polynomial(Falloff):
    """Polynomial decay: target_dot = original_dot * (decay ** power)"""

    def __init__(self, decay: float = 0.5, power: float = 2.0):
        assert 0 <= decay <= 1, 'decay must be in [0, 1]'
        assert power > 0, 'power must be positive'
        self.decay = decay
        self.power = power

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        return dots * (self.decay**self.power)

    def __repr__(self):
        return f'{type(self).__name__}(decay={self.decay}, power={self.power})'


class Exponential(Falloff):
    """Exponential decay: target_dot = original_dot * exp(-strength * (1 - original_dot))"""

    def __init__(self, strength: float = 2.0):
        assert strength > 0, 'strength must be positive'
        self.strength = strength

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        # Only apply to positive dots
        positive_mask = dots > 0
        result = dots.clone()

        if positive_mask.any():
            positive_dots = dots[positive_mask]
            decay = torch.exp(-self.strength * (1 - positive_dots))
            result[positive_mask] = positive_dots * decay

        return result

    def __repr__(self):
        return f'{type(self).__name__}(strength={self.strength})'


class Angular(Falloff):
    """Angular falloff: works in angle space with Gaussian-like falloff"""

    def __init__(self, strength: float = 0.5, width: float = 0.3):
        assert strength > 0, 'push_strength must be positive'
        assert width > 0, 'width must be positive'
        self.strength = strength
        self.width = width

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        # Only apply to positive dots
        positive_mask = dots > 0
        result = dots.clone()

        if positive_mask.any():
            positive_dots = torch.clamp(dots[positive_mask], -1 + 1e-8, 1 - 1e-8)

            # Convert to angles
            original_angles = torch.acos(positive_dots)

            # Apply angular push with exponential falloff
            angle_push = self.strength * torch.exp(-original_angles / self.width)
            new_angles = original_angles + angle_push

            # Convert back to dot products, ensuring we stay in valid range
            new_angles = torch.clamp(new_angles, 0, torch.pi)
            result[positive_mask] = torch.cos(new_angles)

        return result

    def __repr__(self):
        return f'{type(self).__name__}(strength={self.strength}, width={self.width})'


class Sigmoid(Falloff):
    """Sigmoid falloff: smooth transition with configurable center and steepness"""

    def __init__(self, center: float = 0.5, steepness: float = 5.0, min_decay: float = 0.1):
        assert -1 <= center <= 1, 'center must be in [-1, 1]'
        assert steepness > 0, 'steepness must be positive'
        assert 0 <= min_decay <= 1, 'min_decay must be in [0, 1]'
        self.center = center
        self.steepness = steepness
        self.min_decay = min_decay

    @override
    def __call__(self, dots: Tensor) -> Tensor:
        sigmoid = 1 / (1 + torch.exp(self.steepness * (dots - self.center)))
        decay = self.min_decay + (1 - self.min_decay) * sigmoid
        return dots * decay

    def __repr__(self):
        return f'{type(self).__name__}(center={self.center}, steepness={self.steepness}, min_decay={self.min_decay})'
