import torch
from torch import Tensor

from ex_color.loss.regularizer import Regularizer


class AxisAlignedSubspace(Regularizer):
    """Encourages activations to stay within a nominated set of dimensions."""

    def __init__(self, dims: tuple[int, ...], invert=False) -> None:
        """
        Initialize regularizer for a target subspace.

        Args:
            dims: Indices of activation features to keep (allowed to be non-zero).
                All other feature dimensions are penalized (L2) toward zero.
                Default keeps the first two dims, matching previous behavior.
            invert: If True, penalize the specified dims instead of keeping them.
        """
        super().__init__()
        # Normalize + validate
        if len(dims) == 0:
            msg = 'dims must contain at least one index'
            raise ValueError(msg)
        if len(set(dims)) != len(dims):
            msg = 'dims must not contain duplicates'
            raise ValueError(msg)
        if any(i < 0 for i in dims):
            msg = 'dims must be non-negative'
            raise ValueError(msg)
        self.dims = tuple(sorted(dims))
        self.invert = invert

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f'{self.__class__.__name__}(dims={self.dims})'

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
        if not self.invert:
            effective_keep = tuple(i for i in range(F) if i in self.dims)
        else:
            effective_keep = tuple(i for i in range(F) if i not in self.dims)
        # effective_keep = {i for i in self.dims if i < F}
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


class Planarity(AxisAlignedSubspace):
    """Encourages activations to stay within two dimensions."""

    def __init__(self, dims: tuple[int, int] = (0, 1)):
        super().__init__(dims)
