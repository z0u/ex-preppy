import logging
from typing import override

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class ColorMLP(nn.Module):
    """Simple RGB-to-RGB bottlenecked autoencoder"""

    def __init__(self, k_bottleneck: int):
        super().__init__()
        # RGB input (3D) → hidden layer → bottleneck → hidden layer → RGB output
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, k_bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(k_bottleneck, 16),
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


class CNColorMLP(nn.Module):
    """A clamped-and-normalized RGB-to-RGB bottlenecked autoencoder"""

    def __init__(self, k_bottleneck: int, k_codec: int = 16, n_nonlinear: int = 1):
        super().__init__()
        # RGB input (3D) → hidden layer → bottleneck → hidden layer → RGB output
        self.encoder = nn.Sequential(
            nn.Linear(3, k_codec),
            nn.GELU(),
            # Plus zero or more additional nonlinear
            *(m for _ in range(n_nonlinear - 1) for m in (nn.Linear(k_codec, k_codec), nn.GELU())),
            nn.Linear(k_codec, k_bottleneck),
        )

        self.bottleneck = L2Norm()

        self.decoder = nn.Sequential(
            nn.Linear(k_bottleneck, k_codec),
            nn.GELU(),
            # Plus zero or more additional nonlinear
            *(m for _ in range(n_nonlinear - 1) for m in (nn.Linear(k_codec, k_codec), nn.GELU())),
            nn.Linear(k_codec, 3),
        )

    @override  # Overridden to narrow types
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Get the bottleneck representation (captured by a hook for regularization)
        x = self.encoder(x)

        # Normalize
        x = self.bottleneck(x)

        # Decode back to RGB
        x = self.decoder(x)
        return x if self.training else torch.clamp(x, 0, 1)


class L2Norm(nn.Module):
    def forward(self, x: Tensor):
        return nn.functional.normalize(x, dim=-1)
