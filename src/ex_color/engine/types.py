from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass
class InferenceResult:
    outputs: Tensor
    latents: Tensor

    def detach(self):
        return InferenceResult(self.outputs.detach(), self.latents.detach())

    def clone(self):
        return InferenceResult(self.outputs.clone(), self.latents.clone())

    def cpu(self):
        return InferenceResult(self.outputs.cpu(), self.latents.cpu())


@runtime_checkable
class LossCriterion(Protocol):
    def __call__(self, data: Tensor, res: InferenceResult) -> Tensor: ...


@runtime_checkable
class SpecialLossCriterion(LossCriterion, Protocol):
    # Allow returning None if the criterion doesn't apply (e.g., Anchor before anchoring)
    def forward(self, model: 'torch.nn.Module', data: Tensor) -> InferenceResult | None: ...
