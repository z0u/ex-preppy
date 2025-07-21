from typing import Protocol

from torch import Tensor


class Objective[T](Protocol):
    def __call__(self, y_pred: T, y_true: T) -> Tensor: ...
