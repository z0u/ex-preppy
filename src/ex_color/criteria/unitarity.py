from torch import Tensor
from torch import linalg as LA

from ex_color.result import InferenceResult


def unitarity(data: Tensor, res: InferenceResult) -> Tensor:
    """Regularize latents to have unit norm (vectors of length 1)"""
    norms = LA.vector_norm(res.latents, dim=-1)
    # Return per-sample loss, shape [B]
    return (norms - 1.0) ** 2
