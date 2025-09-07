import logging
from typing import cast

import torch
from torch import Tensor

from ex_color.loss.regularizer import Regularizer

log = logging.getLogger(__name__)


class Anchor(Regularizer):
    anchor_point: Tensor

    def __init__(self, anchor_point: Tensor):
        super().__init__()
        self.register_buffer('anchor_point', anchor_point)

    def __call__(self, activations: Tensor) -> Tensor:
        """Regularize latents to be close to the anchor point."""
        return torch.sum((activations - self.anchor_point) ** 2, dim=-1)  # [B]

    def __repr__(self):  # pragma: no cover - simple
        return f'{type(self).__name__}({repr(self.anchor_point)})'


class _AngularPointRegularizer(Regularizer):
    """Base for cosine-sim regularizers around a direction."""

    anchor_point: Tensor

    def __init__(self, anchor_point: Tensor):
        super().__init__()
        self.register_buffer('anchor_point', anchor_point)
        unit = anchor_point / (anchor_point.norm() + 1e-8)
        self.register_buffer('anchor_unit', unit)

    def _cos(self, activations: Tensor) -> Tensor:
        acts_unit = activations / (activations.norm(dim=-1, keepdim=True) + 1e-8)
        anchor_unit = cast(Tensor, self.anchor_unit)
        return torch.sum(acts_unit * anchor_unit, dim=-1)

    def __repr__(self):  # pragma: no cover - simple
        return f'{type(self).__name__}(anchor_point={repr(self.anchor_point)})'


class AngularAnchor(_AngularPointRegularizer):
    """
    Attract activations toward an anchor direction over full sphere.

    Loss = 1 - cos(sim). Aligned -> 0; opposite -> 2.
    """

    def __init__(self, anchor_point: Tensor):
        super().__init__(anchor_point)

    def __call__(self, activations: Tensor) -> Tensor:
        return 1.0 - self._cos(activations)


class AntiAnchor(_AngularPointRegularizer):
    """
    Repel activations from an anchor direction.

    Args:
        hemi: If True, only penalize the forward hemisphere; loss = max(cos, 0).
              If False, map cosine similarity from [-1,1] to [0,1] linearly: (cos+1)/2.
    """

    def __init__(self, anchor_point: Tensor, *, hemi: bool = True):
        super().__init__(anchor_point)
        self.hemi = hemi

    def __call__(self, activations: Tensor) -> Tensor:
        cos = self._cos(activations)
        if self.hemi:
            # cos<=0 => 0 ; cos==1 =>1
            return torch.clamp(cos, min=0.0)
        # full sphere: -1->0, 0->0.5, 1->1
        return (cos + 1.0) / 2.0

    def __repr__(self):  # pragma: no cover - simple
        return f'{type(self).__name__}(anchor_point={repr(self.anchor_point)}, hemi={self.hemi})'
