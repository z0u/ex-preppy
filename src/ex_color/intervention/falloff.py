from typing import override

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
