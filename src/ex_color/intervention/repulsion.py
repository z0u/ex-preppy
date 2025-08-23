from typing import override

import torch
from torch import Tensor

from ex_color.intervention.intervention import Intervention, Mapper, VarAnnotation


class Repulsion(Intervention):
    kind = 'rotational'

    def __init__(
        self,
        concept_vector: Tensor,  # Embedding to steer away from [E] (unit norm)
        mapper: Mapper,  # Function to recalculate dot products to determine rotation of activations
        eps: float = 1e-8,  # Numerical stability threshold
    ):
        """
        Repel activations away from subject vector by rotating in their shared plane.

        Returns:
            Rotated activations with unit norm, shape [B, E].
        """
        super().__init__(concept_vector)
        self.mapper = mapper
        self.eps = eps

    @override
    def dist(self, activations: Tensor) -> Tensor:
        dots = torch.sum(activations * self.concept_vector[None, :], dim=1)  # [B]
        return torch.clamp(dots, 0, 1)

    @override
    def forward(self, activations: Tensor) -> Tensor:
        # Calculate original dot products
        dots = self.dist(activations)  # [B]

        # Scale dot products with falloff function
        target_dots = self.mapper(dots)  # [B]

        # Decompose into parallel and perpendicular components
        v_parallel = dots[:, None] * self.concept_vector[None, :]  # [B, E]
        v_perp = activations - v_parallel  # [B, E]

        # Get perpendicular unit vectors (handle near-parallel case)
        v_perp_norm = torch.norm(v_perp, dim=1, keepdim=True)  # [B, 1]

        # For nearly parallel vectors, choose random orthogonal direction
        nearly_parallel = (v_perp_norm < self.eps).squeeze()  # [B]

        if nearly_parallel.any():
            # Generate random orthogonal vectors
            random_vecs = torch.randn_like(v_perp[nearly_parallel])
            # Make orthogonal to subject using Gram-Schmidt
            proj = torch.sum(random_vecs * self.concept_vector[None, :], dim=1, keepdim=True)
            random_vecs = random_vecs - proj * self.concept_vector[None, :]
            random_vecs = random_vecs / torch.norm(random_vecs, dim=1, keepdim=True)

            v_perp[nearly_parallel] = random_vecs
            v_perp_norm[nearly_parallel] = 1.0

        u_perp = v_perp / v_perp_norm  # [B, E]

        # Construct rotated vectors in the (subject, u_perp) plane
        target_dots_clamped = torch.clamp(target_dots, -1 + self.eps, 1 - self.eps)
        perp_component = torch.sqrt(1 - target_dots_clamped**2)  # [B]

        v_rotated = (
            target_dots_clamped[:, None] * self.concept_vector[None, :] + perp_component[:, None] * u_perp
        )  # [B, E]

        # Only apply rotation to vectors with positive original dot product
        should_rotate = dots > 0  # [B]
        return torch.where(should_rotate[:, None], v_rotated, activations)

    @property
    @override
    def annotations(self):
        return self.mapper.annotations

    @override
    def annotate_activations(self, activations: Tensor):
        dots = self.dist(activations)  # [B]
        target_dots = self.mapper(dots)  # [B]
        return VarAnnotation('offset', (dots - target_dots).abs())

    def __repr__(self):
        return f'{type(self).__name__}({self.concept_vector:r}, {self.falloff:r})'

    def __str__(self):
        return f'repel from {self.concept_vector.tolist()} as {self.mapper}'
