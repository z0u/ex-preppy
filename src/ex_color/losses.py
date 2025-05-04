import logging

import torch
import torch.nn as nn
from torch import linalg as LA
from torch import Tensor

from ex_color.engine.events import Event
from ex_color.engine.types import InferenceResult, LossCriterion, SpecialLossCriterion

log = logging.getLogger(__name__)


def objective(fn):
    """Adapt loss function to look like a regularizer"""

    def wrapper(data: Tensor, res: InferenceResult) -> Tensor:
        # Assume the original function takes (data, outputs)
        return fn(data, res.outputs)

    return wrapper


def unitary(data: Tensor, res: InferenceResult) -> Tensor:
    """Regularize latents to have unit norm (vectors of length 1)"""
    norms = LA.vector_norm(res.latents, dim=-1)
    return torch.mean((norms - 1.0) ** 2)


def planarity(data: Tensor, res: InferenceResult) -> Tensor:
    """Regularize latents to be planar in the first two channels (so zero in other channels)"""
    return torch.mean(res.latents[:, 2:] ** 2)


class Separate(LossCriterion):
    def __init__(self, channels: tuple[int, ...] = (0, 1)):
        self.channels = channels

    def __call__(self, data: Tensor, res: InferenceResult) -> Tensor:
        """Regularize latents to be separated from each other in specified channels"""
        # Get pairwise differences in the specified dimensions
        points = res.latents[:, self.channels]  # [B, C]
        diffs = points.unsqueeze(1) - points.unsqueeze(0)  # [B, B, C]

        # Calculate squared distances
        sq_dists = torch.sum(diffs**2, dim=-1)  # [B, B]

        # Remove self-distances (diagonal)
        mask = 1.0 - torch.eye(sq_dists.shape[0], device=sq_dists.device)
        masked_sq_dists = sq_dists * mask

        # Encourage separation by minimizing inverse distances (stronger repulsion between close points)
        epsilon = 1e-6  # Prevent division by zero
        # Use mean of non-zero elements to avoid bias from batch size
        non_zero_elements = mask.sum()
        if non_zero_elements == 0:
            return torch.tensor(0.0, device=sq_dists.device)
        return torch.sum(1.0 / (masked_sq_dists + epsilon)) / non_zero_elements


class Anchor(SpecialLossCriterion):
    """Regularize latents to be close to their position in the reference phase"""

    ref_data: Tensor
    _ref_latents: Tensor | None = None

    def __init__(self, ref_data: Tensor):
        self.ref_data = ref_data
        self._ref_latents = None
        log.info(f'Anchor initialized with reference data shape: {ref_data.shape}')

    def forward(self, model: nn.Module, data: Tensor) -> InferenceResult | None:
        """Run the *stored reference data* through the *current* model."""
        # Note: The 'data' argument passed by the training loop for SpecialLossCriterion
        # is the *current training batch*, which we IGNORE here.
        # We only care about running our stored _ref_data through the model.
        device = next(model.parameters()).device
        ref_data = self.ref_data.to(device)

        # Need to handle the model output structure correctly.
        # Assuming the model returns (output, latents) tuple like ColorMLP
        # If the model type is generic nn.Module, we might need a different way
        # or assume a specific interface. Let's assume the tuple for now.
        model_output = model(ref_data)
        if not (isinstance(model_output, tuple) and len(model_output) == 2):
            raise TypeError(f'Anchor regularizer expects model to return (outputs, latents), got {type(model_output)}')
        outputs, latents = model_output

        return InferenceResult(outputs, latents)

    def __call__(self, data: Tensor, special: InferenceResult) -> Tensor:
        """Calculates loss between current model's latents (for ref_data) and the stored reference latents."""
        if self._ref_latents is None:
            # This means on_anchor hasn't been called yet, so the anchor loss is zero.
            # This prevents errors during the very first phase before the anchor point is set.
            log.debug('Anchor.__call__ invoked before reference latents captured. Returning zero loss.')
            return torch.tensor(0.0, device=special.latents.device)
        ref_latents = self._ref_latents.to(special.latents.device)
        return torch.mean((special.latents - ref_latents) ** 2)

    def on_anchor(self, event: Event):
        # Called when the 'anchor' event is triggered
        log.info(f'Capturing anchor latents via Anchor.on_anchor at step {event.step}')

        device = next(event.model.parameters()).device
        ref_data = self.ref_data.to(device)

        with torch.no_grad():
            # Assuming the model returns (output, latents) tuple
            model_output = event.model(ref_data)
            if not (isinstance(model_output, tuple) and len(model_output) == 2):
                raise TypeError(
                    f'Anchor regularizer expects model to return (outputs, latents), got {type(model_output)}'
                )
            _, latents = model_output

        self._ref_latents = latents.detach().cpu()
        log.info(f'Anchor state captured internally. Ref data: {ref_data.shape}, Ref latents: {latents.shape}')
