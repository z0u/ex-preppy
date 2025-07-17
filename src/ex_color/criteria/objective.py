import torch
from torch import Tensor

from ex_color.result import InferenceResult


def objective(fn):
    """Adapt loss function to look like a regularizer"""

    def wrapper(data: Tensor, res: InferenceResult) -> Tensor:
        loss = fn(data, res.outputs)
        # Reduce element-wise loss to per-sample loss by averaging over feature dimensions
        if loss.ndim > 1:
            # Calculate mean over all dimensions except the first (batch) dimension
            reduce_dims = tuple(range(1, loss.ndim))
            loss = torch.mean(loss, dim=reduce_dims)
        return loss

    return wrapper
