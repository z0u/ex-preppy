from typing import override

import torch
from torch import Tensor

from ex_color.intervention.intervention import Falloff, Intervention


class Repulsion(Intervention):
    def __init__(
        self,
        subject: Tensor,  # Embedding to steer away from [E] (unit norm)
        falloff: Falloff,  # Function to recalculate dot products to determine rotation of activations
        eps: float = 1e-8,  # Numerical stability threshold
    ):
        """
        Repel activations away from subject vector by rotating in their shared plane.

        Returns:
            Rotated activations with unit norm, shape [B, E].
        """
        super().__init__(subject, falloff)
        self.eps = eps

    def dist(self, activations: Tensor) -> Tensor:
        dots = torch.sum(activations * self.subject[None, :], dim=1)  # [B]
        return torch.clamp(dots, 0, 1)

    @override
    def __call__(self, activations: Tensor) -> Tensor:
        # Calculate original dot products
        dots = torch.sum(activations * self.subject[None, :], dim=1)  # [B]

        # Scale dot products with falloff function
        target_dots = self.falloff(dots)  # [B]

        # Decompose into parallel and perpendicular components
        v_parallel = dots[:, None] * self.subject[None, :]  # [B, E]
        v_perp = activations - v_parallel  # [B, E]

        # Get perpendicular unit vectors (handle near-parallel case)
        v_perp_norm = torch.norm(v_perp, dim=1, keepdim=True)  # [B, 1]

        # For nearly parallel vectors, choose random orthogonal direction
        nearly_parallel = (v_perp_norm < self.eps).squeeze()  # [B]

        if nearly_parallel.any():
            # Generate random orthogonal vectors
            random_vecs = torch.randn_like(v_perp[nearly_parallel])
            # Make orthogonal to subject using Gram-Schmidt
            proj = torch.sum(random_vecs * self.subject[None, :], dim=1, keepdim=True)
            random_vecs = random_vecs - proj * self.subject[None, :]
            random_vecs = random_vecs / torch.norm(random_vecs, dim=1, keepdim=True)

            v_perp[nearly_parallel] = random_vecs
            v_perp_norm[nearly_parallel] = 1.0

        u_perp = v_perp / v_perp_norm  # [B, E]

        # Construct rotated vectors in the (subject, u_perp) plane
        target_dots_clamped = torch.clamp(target_dots, -1 + self.eps, 1 - self.eps)
        perp_component = torch.sqrt(1 - target_dots_clamped**2)  # [B]

        v_rotated = target_dots_clamped[:, None] * self.subject[None, :] + perp_component[:, None] * u_perp  # [B, E]

        # Only apply rotation to vectors with positive original dot product
        should_rotate = dots > 0  # [B]
        return torch.where(should_rotate[:, None], v_rotated, activations)
