"""External hook management for latent capture."""

import torch
import torch.nn as nn


class LatentHookManager:
    """External manager for capturing latents from model layers."""
    
    def __init__(self):
        self._latents: torch.Tensor | None = None
        self._hook_handle: torch.utils.hooks.RemovableHandle | None = None
    
    def register_hook(self, layer: nn.Module):
        """Register a hook to capture latents from the specified layer."""
        
        def hook_fn(module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:  # noqa: ARG001
            del module, input  # Unused parameters required by hook signature
            self._latents = output
        
        # Register hook on the layer
        self._hook_handle = layer.register_forward_hook(hook_fn)
    
    def remove_hook(self):
        """Remove the latent hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            self._latents = None
    
    def get_latents(self) -> torch.Tensor | None:
        """Get the last captured latents."""
        return self._latents
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures hook cleanup."""
        self.remove_hook()