import logging
from typing import override

import torch.nn as nn
from torch import Tensor

E = 4

log = logging.getLogger(__name__)


class ColorMLP(nn.Module):
    """Pure neural network model for color transformation."""

    def __init__(self):
        super().__init__()
        # RGB input (3D) → hidden layer → bottleneck → hidden layer → RGB output
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, E),  # Our critical bottleneck!
        )

        self.decoder = nn.Sequential(
            nn.Linear(E, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),  # Keep RGB values in [0,1]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Get our bottleneck representation
        latents = self.encoder(x)

        # Decode back to RGB
        return self.decoder(latents)
