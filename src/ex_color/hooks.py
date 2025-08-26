import logging

import torch
from torch import Tensor

from ex_color.intervention.intervention import InterventionConfig

log = logging.getLogger(__name__)


class ActivationCaptureHook:
    """
    Captures latent representations, i.e. layer activations.

    See:
    - `ActivationModifyHook`
    """

    def __init__(self):
        self.activations: Tensor | None = None

    def __call__(self, _module, _input, output: Tensor):
        self.activations = output


class ActivationCaptureBufferHook:
    """
    Captures latent representations, storing them in a buffer.

    This hook collects activations over several batches; they can then be
    retrieved by calling `read()`. Since operation typically spans multiple
    steps, the activation tensors are always detached and moved to the CPU.
    """

    def __init__(self):
        self.batches: list[Tensor] = []

    def __call__(self, _module, _input, output: Tensor):
        self.batches.append(output.detach().cpu().clone())

    def read(self):
        """
        Consume activations as a tensor and clear the buffer.

        Returns: the activations collected since the last call to `read`, as a
        detached tensor on the CPU with shape [B*steps,...].
        """
        activations = torch.cat(self.batches)
        self.batches = []
        return activations


class ActivationModifyHook:
    """
    Modifies latent representations, i.e. layer activations.

    See:
    - `ActivationCaptureHook`
    """

    def __init__(self, intervention: InterventionConfig):
        self.intervention = intervention

    def __call__(self, module, _input, output: Tensor) -> Tensor:
        return self.intervention.apply(output)
