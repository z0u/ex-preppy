from typing import Annotated, override

from pydantic import Field, validate_call
from torch import Tensor

ZeroToOne = Annotated[float, Field(ge=0, le=1)]


class Falloff:
    """
    Determine intervention amount based on distance to subject.

    Args:
        alignment: Closeness of activations from the subject, of shape [B],
        where 1 is "perfectly aligned" and 0 is "not at all aligned". This could
        be cosine distance or some other measure.

    Returns:
        The offset from the original distance, of shape [B].
    """

    def __call__(self, alignment: Tensor, /) -> Tensor: ...


class Linear(Falloff):
    """
    Linear decay: d.decay

    For cosine distance: produces a circular lobe between 0 and `decay`.
    """

    @validate_call
    def __init__(self, decay: ZeroToOne):
        self.decay = decay

    @override
    def __call__(self, alignment: Tensor) -> Tensor:
        return alignment * self.decay

    def __repr__(self):
        return f'{type(self).__name__}(decay={self.decay:.2g})'

    def __str__(self):
        return f'd*{self.decay:.2g}'


class Power(Falloff):
    """
    Power falloff: d^power

    Produces a specular-like lobe.
    """

    @validate_call
    def __init__(self, power: float = 10):
        self.power = power

    @override
    def __call__(self, alignment: Tensor) -> Tensor:
        return alignment**self.power

    def __repr__(self):
        return f'{type(self).__name__}({self.power})'

    def __str__(self):
        return f'd^{self.power:.2g}'
