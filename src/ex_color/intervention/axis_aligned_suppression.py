"""Zero out specific activation dimensions (axis-aligned suppression)."""

from typing import override

import torch
from torch import Tensor

from ex_color.intervention.intervention import Intervention, VarAnnotation


class AxisAlignedSuppression(Intervention):
    """Suppress activations by zeroing specific dimensions.

    Unlike directional suppression (Suppression), this operates on axis-aligned
    subspaces by simply zeroing out the specified dimensions.
    """

    kind = 'linear'
    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]):
        """Initialize the intervention.

        Args:
            dims: Dimensions to zero out (e.g., (0, 1) for first two dimensions)
        """
        super().__init__()
        self.dims = dims

    @override
    def dist(self, activations: Tensor) -> Tensor:
        """Calculate the L2 norm of components in the suppressed dimensions.

        Args:
            activations: Shape [B, E] where E is embedding dimension

        Returns:
            Distance tensor of shape [B]
        """
        # Sum of squares of the dimensions to be suppressed
        return torch.sqrt(torch.sum(activations[:, self.dims] ** 2, dim=1))

    @override
    def forward(self, activations: Tensor) -> Tensor:
        """Zero out the specified dimensions.

        Args:
            activations: Shape [B, E] where E is embedding dimension

        Returns:
            Modified activations with specified dimensions zeroed
        """
        result = activations.clone()
        result[:, self.dims] = 0
        return result

    @property
    @override
    def annotations(self):
        return []

    @override
    def annotate_activations(self, activations: Tensor):
        return VarAnnotation('suppressed_norm', self.dist(activations))

    def __repr__(self):
        return f'{type(self).__name__}(dims={self.dims})'

    def __str__(self):
        return f'zero dimensions {self.dims}'
