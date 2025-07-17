import logging
from dataclasses import dataclass

from torch import Tensor


log = logging.getLogger(__name__)


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
