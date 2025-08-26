from torch import Tensor
from torch import linalg as LA

from ex_color.loss.regularizer import Regularizer


class Unitarity(Regularizer):
    def __call__(self, activations: Tensor) -> Tensor:
        """Regularize latents to have unit norm (vectors of length 1)"""
        norms = LA.vector_norm(activations, dim=-1)
        # Return per-sample loss, shape [B]
        return (norms - 1.0) ** 2
