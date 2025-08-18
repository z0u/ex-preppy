from typing import override

import torch
from torch import Tensor

from ex_color.intervention.intervention import Falloff, Intervention


class Suppression(Intervention):
    type = 'linear'

    def __init__(
        self,
        subject: Tensor,  # Embedding to steer away from [E] (unit norm)
        falloff: Falloff,  # Function to recalculate dot products to determine rotation of activations
        *,
        amount: float = 1.0,  # Numerical stability threshold
        renormalize=False,  # False -> squash, True -> redirect
        bidirectional=False,  # If False, only positively-aligned embeddings will be modified. In either case, orthogonal components will be untouched.
    ):
        """
        Repel activations away from subject vector by rotating in their shared plane.

        Returns:
            Rotated activations with unit norm, shape [B, E].
        """
        super().__init__(subject, falloff)
        self.amount = amount
        self.renormalize = renormalize
        self.bidirectional = bidirectional

    @override
    def dist(self, activations: Tensor) -> Tensor:
        dots = torch.sum(activations * self.subject[None, :], dim=1)  # [B]
        if self.bidirectional:
            dots = dots.abs()
        dots = dots.clamp(min=0, max=1)
        return dots

    @override
    def __call__(self, activations: Tensor) -> Tensor:
        # v: Tensor = self.subject / torch.linalg.norm(self.subject)
        # a_norm: Tensor = activations / (activations.norm(dim=-1, keepdim=True) + 1e-12)

        cos_sim = self.dist(activations)
        # cos_sim = torch.einsum('b...e,e->b...', a_norm, v)
        # if self.bidirectional:
        #     cos_sim = cos_sim.abs()
        # cos_sim = cos_sim.clamp(min=0, max=1)

        gate = self.falloff(cos_sim) * self.amount
        p = torch.einsum('b...e,e->b...', activations, self.subject)

        out = activations - torch.einsum('b...,e->b...e', gate * p, self.subject)
        if self.renormalize:
            norms = activations.norm(dim=-1, keepdim=True) + 1e-12
            out = out / out.norm(dim=-1, keepdim=True) * norms
        return out
