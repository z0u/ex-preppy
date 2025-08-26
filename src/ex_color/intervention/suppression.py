from typing import override

import torch
from torch import Tensor

from ex_color.intervention.intervention import Intervention, Mapper, VarAnnotation


class Suppression(Intervention):
    kind = 'linear'
    concept_vector: Tensor

    def __init__(
        self,
        concept_vector: Tensor,  # Embedding to suppress [E] (unit norm)
        falloff: Mapper,  # Function to calculate strength of suppression
    ):
        super().__init__()
        self.falloff = falloff
        # Register as buffer so it's saved/moved with the module but not trained
        self.register_buffer('concept_vector', concept_vector)

    @override
    def dist(self, activations: Tensor) -> Tensor:
        dots = torch.sum(activations * self.concept_vector[None, :], dim=1)  # [B]
        return dots.clamp(min=0, max=1)

    def gate(self, activations: Tensor) -> Tensor:
        return self.falloff(self.dist(activations))

    @override
    def forward(self, activations: Tensor) -> Tensor:
        self.concept_vector = self.concept_vector.to(activations)
        gate = self.gate(activations)
        p = torch.einsum('b...e,e->b...', activations, self.concept_vector)
        return activations - torch.einsum('b...,e->b...e', gate * p, self.concept_vector)

    @property
    @override
    def annotations(self):
        return self.falloff.annotations

    @override
    def annotate_activations(self, activations: Tensor):
        return VarAnnotation('strength', self.gate(activations))

    def __repr__(self):
        return f'{type(self).__name__}({self.concept_vector:r}, {self.falloff:r})'

    def __str__(self):
        return f'suppress {self.concept_vector.tolist()} as {self.falloff}'
