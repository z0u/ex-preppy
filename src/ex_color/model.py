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

        # Hook storage for latents
        self._latents: torch.Tensor | None = None
        self._hook_handle: torch.utils.hooks.RemovableHandle | None = None

    def register_latent_hook(self):
        """Register a hook to capture bottleneck latents."""

        def hook_fn(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:  # noqa: ARG001
            self._latents = output

        # Register hook on the encoder's last layer (bottleneck)
        self._hook_handle = self.encoder[-1].register_forward_hook(hook_fn)

    def remove_latent_hook(self):
        """Remove the latent hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_latents(self) -> torch.Tensor | None:
        """Get the last captured latents."""
        return self._latents

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get our bottleneck representation
        bottleneck = self.encoder(x)

        # Decode back to RGB
        output = self.decoder(bottleneck)
        return output
