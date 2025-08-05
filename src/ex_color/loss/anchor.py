import logging

import torch
from torch import Tensor

from ex_color.loss.regularizer import Regularizer

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

    def __repr__(self):
        # for experiment trackers/loggers like wandb
        return f'{type(self).__name__}({repr(self.anchor_point)})'
