import logging
from typing import override

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from ex_color.intervention.intervention import InterventionConfig

log = logging.getLogger(__name__)


class ActivationModifyHook:
    """
    Modifies latent representations, i.e. layer activations.

    See:
    - `ActivationCaptureHook`
    """

    def __init__(self, intervention: InterventionConfig):
        self.intervention = intervention

    def __call__(self, module, _input, output):
        if self.intervention.strength == 0:
            return output

        modified_output = self.intervention.apply(output)

        if self.intervention.strength == 1:
            return modified_output

        return torch.lerp(output, modified_output, self.intervention.strength)


class InferenceModule(L.LightningModule):
    """Lightning module that handles inference logic for any model architecture."""

    def __init__(
        self,
        model: nn.Module,
        interventions: list[InterventionConfig],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.interventions = interventions
        self.hook_handles: list[RemovableHandle] = []

    def _setup_hooks(self):
        """Register each intervention as its own forward hook."""
        # PyTorch will chain them automatically
        for intervention in self.interventions:
            for layer_name in intervention.layer_affinities:
                try:
                    layer_module = self.model.get_submodule(layer_name)
                except AttributeError as e:
                    raise AttributeError(f'Layer {layer_name} (needed by {intervention.name})') from e

                hook = ActivationModifyHook(intervention)
                handle = layer_module.register_forward_hook(hook)
                self.hook_handles.append(handle)
                log.debug(
                    f'Registered intervention hook for layer: {layer_name} ("{intervention.name}" {repr(intervention.apply)})'
                )

    @override
    def on_predict_start(self):
        """Called at the very beginning of predict. Set up hooks here for DDP compatibility."""
        super().on_predict_start()
        self._setup_hooks()

    @override
    def on_predict_end(self):
        """Called at the very end of predict. Clean up hooks."""
        super().on_predict_end()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @override
    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        return self(batch)
