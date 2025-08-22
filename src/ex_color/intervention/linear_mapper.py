from typing import Annotated, override

import torch
from annotated_types import Ge, Gt, Le, Lt
from pydantic import validate_call
from torch import Tensor

from ex_color.intervention.intervention import ConstAnnotation, Mapper


class LinearMapper(Mapper):
    @validate_call
    def __init__(
        self,
        a: Annotated[float, [Ge(0), Lt(1)]],
        b: Annotated[float, [Gt(0), Le(1)]],
        eps=1e-8,
    ):
        super().__init__()
        assert a < b
        self.a = a
        self.b = b
        self.eps = eps

    @override
    def forward(
        self,
        alignment: Tensor,  # [B]
    ) -> Tensor:  # [B]
        shifted = (alignment - self.a) / (1 - self.a)
        shifted = shifted * (self.b - self.a) + self.a
        return torch.where(alignment > self.a, shifted, alignment)

    @property
    @override
    def annotations(self):
        return [
            ConstAnnotation('input', 'angular', 'a (start)', self.a),
            ConstAnnotation('output', 'angular', 'b (end)', self.b),
        ]

    def __repr__(self):
        return f'{type(self).__name__}({self.a:.2g}, {self.b:.2g})'

    def __str__(self):
        components = ['Linear']
        if self.a != 0:
            components.append(rf'$a = {self.a:.2g}$')
        if self.b != 1:
            components.append(rf'$b = {self.b:.2g}$')
        return rf'{", ".join(components)}'
