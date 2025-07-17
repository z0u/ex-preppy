import logging

import torch
import torch.nn as nn

E = 4

log = logging.getLogger(__name__)


class ColorMLP(nn.Module):
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



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get our bottleneck representation
        bottleneck = self.encoder(x)

        # Decode back to RGB
        output = self.decoder(bottleneck)
        return output
