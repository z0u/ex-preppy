import logging
from types import EllipsisType

import torch
from torch import Tensor

from ex_color.criteria.criteria import LossCriterion
from ex_color.result import InferenceResult

log = logging.getLogger(__name__)


class Separate(LossCriterion):
    """Regularize latents to be rotationally separated from each other."""

    def __init__(self, channels: tuple[int, ...] | EllipsisType = ..., power: float = 1.0, shift: bool = True):
        self.channels = channels
        self.power = power
        self.shift = shift

    def __call__(self, data: Tensor, res: InferenceResult) -> Tensor:
        embeddings = res.latents[:, self.channels]  # [B, C]

        # Normalize to unit hypersphere, so it's only the angular distance that matters
        embeddings = embeddings / (torch.norm(embeddings, dim=-1, keepdim=True) + 1e-8)

        # Find the angular distance as cosine similarity
        cos_sim = torch.matmul(embeddings, embeddings.T)  # [B, B]

        # Nullify self-repulsion.
        # We can't use torch.eye, because some points in the batch may be duplicates due to the use of random sampling with replacement.
        cos_sim[torch.isclose(cos_sim, torch.ones_like(cos_sim))] = 0.0
        if self.shift:
            # Shift the cosine similarity to be in the range [0, 1]
            cos_sim = (cos_sim + 1.0) / 2.0

        # Sum over all other points
        return torch.sum(cos_sim**self.power, dim=-1)  # [B]
