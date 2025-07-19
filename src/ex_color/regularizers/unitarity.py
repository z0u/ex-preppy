from torch import Tensor
from torch import linalg as LA


def unitarity(activations: Tensor) -> Tensor:
    """Regularize latents to have unit norm (vectors of length 1)"""
    norms = LA.vector_norm(activations, dim=-1)
    # Return per-sample loss, shape [B]
    return (norms - 1.0) ** 2
