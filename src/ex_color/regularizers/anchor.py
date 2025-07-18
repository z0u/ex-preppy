import logging

import torch
from torch import Tensor

from ex_color.regularizers.criteria import Regularizer

log = logging.getLogger(__name__)


class Anchor(Regularizer):
    def __init__(self, anchor_point: Tensor):
        self.anchor_point = anchor_point

    def __call__(self, activations: Tensor) -> Tensor:
        """
        Regularize latents to be close to the anchor point.

        Returns:
            loss: Per-sample loss, shape [B].
        """
        # Calculate squared distances to the anchor
        sq_dists = torch.sum((activations - self.anchor_point) ** 2, dim=-1)  # [B]
        return sq_dists
