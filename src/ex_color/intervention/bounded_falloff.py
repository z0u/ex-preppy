from typing import Annotated, Sequence, override

import torch
from annotated_types import Ge, Le
from pydantic import validate_call
from torch import Tensor

from ex_color.intervention.intervention import ConstAnnotation, Mapper


class BoundedFalloff(Mapper):
    @validate_call
    def __init__(
        self,
        a: Annotated[float, [Ge(0), Le(1)]],
        b: Annotated[float, [Ge(0), Le(1)]],
        power: Annotated[float, [Ge(1)]] = 1.0,
        eps=1e-8,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.power = power
        self.eps = eps

    @override
    def forward(
        self,
        alignment: Tensor,  # [B]
    ) -> Tensor:  # [B]
        if self.a > 1 - self.eps:
            return alignment

        shifted = (alignment - self.a) / (1 - self.a)
        shifted = shifted**self.power * (self.b)
        return torch.where(alignment > self.a, shifted, torch.zeros_like(alignment))

    @property
    @override
    def annotations(self) -> Sequence[ConstAnnotation]:
        return [
            ConstAnnotation('input', 'angular', 'a', self.a),
            ConstAnnotation('output', 'linear', 'b', self.b),
        ]

    def __repr__(self):
        return f'{type(self).__name__}({self.a:.2g}, {self.b:.2g}, {self.power:.2g})'

    def __str__(self):
        components = []
        if self.a != 0:
            components.append(rf'$a={self.a:.2g}$')
        if self.b != 1:
            components.append(rf'$b={self.b:.2g}$')
        if self.power != 1:
            components.append(rf'$p={self.power:.2g}$')
        return ', '.join(components)
