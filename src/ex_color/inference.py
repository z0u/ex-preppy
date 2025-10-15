import logging
from typing import Sequence, override

import lightning as L
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from ex_color.hooks import ActivationCaptureBufferHook, ActivationModifyHook
from ex_color.intervention.intervention import InterventionConfig

log = logging.getLogger(__name__)


class InferenceModule(L.LightningModule):
    """Lightning module that handles inference logic for any model architecture."""

    def __init__(
        self,
        model: nn.Module,
        interventions: Sequence[InterventionConfig],
        capture_layers: Sequence[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.interventions = interventions
        self.hook_handles: list[RemovableHandle] = []
        # Optional activation capture per layer name
        self.capture_layers = capture_layers or []
        self._capture_hooks: dict[str, ActivationCaptureBufferHook] = {}

    def _setup_hooks(self):
        """Register each intervention as its own forward hook."""
        # PyTorch will chain them automatically
        for intervention in self.interventions:
            for layer_name in intervention.layer_affinities:
                try:
                    layer_module = self.model.get_submodule(layer_name)
                except AttributeError as e:
                    raise AttributeError(f'Layer {layer_name} (needed by {intervention.apply})') from e

                hook = ActivationModifyHook(intervention)
                handle = layer_module.register_forward_hook(hook)
                self.hook_handles.append(handle)
                log.debug(f'Registered intervention hook for layer: {layer_name} ({intervention.apply})')

        # Register activation capture hooks AFTER interventions so we capture post-intervention activations
        for layer_name in self.capture_layers:
            try:
                layer_module = self.model.get_submodule(layer_name)
            except AttributeError as e:
                raise AttributeError(f'Layer {layer_name} (requested for activation capture)') from e

            capture_hook = ActivationCaptureBufferHook()
            handle = layer_module.register_forward_hook(capture_hook)
            self.hook_handles.append(handle)
            self._capture_hooks[layer_name] = capture_hook
            log.debug(f'Registered activation capture hook for layer: {layer_name}')

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

    # Do not clear capture buffers here; allow notebook to read them after predict.

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @override
    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        return self(batch)

    # ----- Activation capture API -----
    def read_captured(self, layer_name: str) -> Tensor:
        """
        Consume captured activations for a layer and return as a single tensor on CPU.

        Shape is [N, ...] where N is the total number of items seen across predict batches.
        """
        if layer_name not in self._capture_hooks:
            raise KeyError(f'No capture hook registered for layer: {layer_name}')
        return self._capture_hooks[layer_name].read()
