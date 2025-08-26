import torch
from torch import Tensor

from ex_color.loss.regularizer import Regularizer


class Planarity(Regularizer):
    def __call__(self, activations: Tensor) -> Tensor:
        """Regularize latents to be planar in the first two channels (so zero in other channels)"""
        if activations.shape[1] <= 2:
            # No dimensions beyond the first two, return zero loss per sample
            return torch.zeros(activations.shape[0], device=activations.device)
        # Sum squares across the extra dimensions for each sample, shape [B]
        return torch.sum(activations[:, 2:] ** 2, dim=-1)
