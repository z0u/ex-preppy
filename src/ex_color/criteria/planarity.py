import torch
from torch import Tensor

from ex_color.result import InferenceResult


def planarity(data: Tensor, res: InferenceResult) -> Tensor:
    """Regularize latents to be planar in the first two channels (so zero in other channels)"""
    if res.latents.shape[1] <= 2:
        # No dimensions beyond the first two, return zero loss per sample
        return torch.zeros(res.latents.shape[0], device=res.latents.device)
    # Sum squares across the extra dimensions for each sample, shape [B]
    return torch.sum(res.latents[:, 2:] ** 2, dim=-1)
