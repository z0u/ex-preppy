import logging
from typing import override

import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class ColorMLP(nn.Module):
    """Simple RGB-to-RGB bottlenecked autoencoder"""

    def __init__(self, n_bottleneck: int):
        super().__init__()
        # RGB input (3D) → hidden layer → bottleneck → hidden layer → RGB output
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, n_bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_bottleneck, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),  # Keep RGB values in [0,1]
        )

    @override  # Overridden to narrow types
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Get the bottleneck representation (captured by a hook for regularization)
        x = self.encoder(x)

        # Decode back to RGB
        return self.decoder(x)
