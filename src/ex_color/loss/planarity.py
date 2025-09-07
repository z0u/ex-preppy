import torch
from torch import Tensor

from ex_color.loss.regularizer import Regularizer


class Planarity(Regularizer):
    def __init__(self, keep_indices: tuple[int, ...] = (0, 1)) -> None:
        """
        Initialize planarity regularizer for a target subspace.

        Args:
            keep_indices: Indices of activation features to keep (allowed to be non-zero).
                All other feature dimensions are penalized (L2) toward zero.
                Default keeps the first two dims, matching previous behavior.
        """
        super().__init__()
        # Normalize + validate
        if len(keep_indices) == 0:
            msg = 'keep_indices must contain at least one index'
            raise ValueError(msg)
        if len(set(keep_indices)) != len(keep_indices):
            msg = 'keep_indices must not contain duplicates'
            raise ValueError(msg)
        if any(i < 0 for i in keep_indices):
            msg = 'keep_indices must be non-negative'
            raise ValueError(msg)
        self.keep_indices = tuple(sorted(keep_indices))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'{self.__class__.__name__}(keep_indices={self.keep_indices})'

    def __call__(self, activations: Tensor) -> Tensor:  # type: ignore[override]
        """
        Regularize latents to lie in the span of selected feature indices.

        Returns per-sample penalty: sum of squared activations for all non-kept dims.
        If activations has fewer feature dims than any keep index, missing kept dims
        implicitly mean they don't exist yet; we only penalize existing non-kept dims.
        """
        if activations.ndim < 2:
            msg = 'activations must have shape [B, F, ...] with at least 2 dims'
            raise ValueError(msg)

        B, F = activations.shape[0], activations.shape[1]
        # Filter keep indices that actually exist in current tensor
        effective_keep = {i for i in self.keep_indices if i < F}
        if len(effective_keep) == F:  # everything kept -> zero penalty
            return torch.zeros(B, device=activations.device, dtype=activations.dtype)

        # Build mask of penalized feature dims
        device = activations.device
        penalize_mask = torch.ones(F, dtype=torch.bool, device=device)
        if effective_keep:
            penalize_mask[list(effective_keep)] = False

        # Select penalized dims and sum squares across those dims + any remaining trailing dims
        # Shape assumptions: activations [B, F, *rest]; we treat F dimension only.
        penalized = activations[:, penalize_mask]
        if penalized.numel() == 0:
            return torch.zeros(B, device=device, dtype=activations.dtype)
        # Sum across feature (and any broadcast leftover) dims except batch
        return torch.sum(penalized * penalized, dim=tuple(range(1, penalized.ndim)))
