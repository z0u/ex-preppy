from math import sqrt
from typing import Annotated, override

import torch
from annotated_types import Ge, Gt, Le, Lt
from pydantic import validate_call
from torch import Tensor

from ex_color.intervention.intervention import ConstAnnotation, Mapper


class BezierMapper(Mapper):
    P0: Tensor
    P1: Tensor
    P2: Tensor
    P3: Tensor

    @validate_call
    def __init__(
        self,
        a: Annotated[float, [Ge(0), Lt(1)]],
        b: Annotated[float, [Gt(0), Le(1)]],
        *,
        start_slope: float = 1.0,  # Aligned with unmapped leadup
        end_slope: float = 0.0,  # Flat
        control_distance: float = 1 / sqrt(2),  # Relative to intersection point
    ):
        super().__init__()
        assert a < b <= 1
        self.a = a
        self.b = b

        # Find intersection of the two tangent lines
        # Line 1: y - a = start_slope * (x - a)  =>  y = start_slope * (x - a) + a
        # Line 2: y - b = end_slope * (x - 1)    =>  y = end_slope * (x - 1) + b
        # At intersection: start_slope * (x - a) + a = end_slope * (x - 1) + b

        if abs(start_slope - end_slope) < 1e-8:
            # Parallel lines - use midpoint as fallback
            intersection_x = (a + 1) / 2
            intersection_y = (a + b) / 2
        else:
            intersection_x = (a * (start_slope - 1) + b - end_slope) / (start_slope - end_slope)
            intersection_y = start_slope * (intersection_x - a) + a

        intersection = torch.tensor([intersection_x, intersection_y], dtype=torch.float32)

        # Define the 4 control points for cubic Bézier
        P0 = torch.tensor([a, a], dtype=torch.float32)
        P3 = torch.tensor([1.0, b], dtype=torch.float32)

        # P1: distance from P0 towards intersection, scaled by control_distance
        direction_to_intersection = intersection - P0
        P1 = P0 + control_distance * direction_to_intersection

        # P2: distance from P3 towards intersection, scaled by control_distance
        direction_to_intersection = intersection - P3
        P2 = P3 + control_distance * direction_to_intersection

        # Register control points as buffers - automatically saved/loaded and moved with module
        self.register_buffer('P0', P0)
        self.register_buffer('P1', P1)
        self.register_buffer('P2', P2)
        self.register_buffer('P3', P3)

    def bezier_point(self, t: Tensor) -> Tensor:
        """Evaluate cubic Bézier curve at parameter t"""
        one_minus_t = 1 - t

        # Buffers are automatically on the correct device
        term0 = (one_minus_t**3)[:, None] * self.P0[None, :]
        term1 = (3 * one_minus_t**2 * t)[:, None] * self.P1[None, :]
        term2 = (3 * one_minus_t * t**2)[:, None] * self.P2[None, :]
        term3 = (t**3)[:, None] * self.P3[None, :]

        return term0 + term1 + term2 + term3

    def bezier_x(self, t: Tensor) -> Tensor:
        """Get x-coordinate of Bézier curve at parameter t"""
        return self.bezier_point(t)[:, 0]

    def bezier_y(self, t: Tensor) -> Tensor:
        """Get y-coordinate of Bézier curve at parameter t"""
        return self.bezier_point(t)[:, 1]

    def solve_for_t(self, x: Tensor, max_iters: int = 10) -> Tensor:
        """Solve for parameter t such that bezier_x(t) = x using Newton's method"""
        # Initial guess: linear interpolation
        t = (x - self.a) / (1 - self.a)
        t = torch.clamp(t, 0.01, 0.99)  # Avoid endpoints

        for _ in range(max_iters):
            # Newton step: t_new = t - f(t)/f'(t)
            # where f(t) = bezier_x(t) - target_x

            # Enable gradients for automatic differentiation
            t_var = t.clone().requires_grad_(True)
            x_pred = self.bezier_x(t_var)
            error = x_pred - x

            # Compute derivative dx/dt
            dx_dt = torch.autograd.grad(x_pred.sum(), t_var, create_graph=False)[0]

            # Newton update (be careful with division by zero)
            dt = error / (dx_dt + 1e-8)
            t = t - dt
            t = torch.clamp(t, 0.0, 1.0)

            # Check convergence
            if torch.max(torch.abs(error)) < 1e-6:
                break

        return t

    @override
    def forward(
        self,
        alignment: Tensor,  # [B]
    ) -> Tensor:  # [B]
        result = alignment.clone()

        # Only apply Bézier mapping for alignment > a
        mask = alignment > self.a
        if mask.any():
            x_vals = alignment[mask]

            # Solve for t parameters
            t_vals = self.solve_for_t(x_vals)

            # Get corresponding y values
            y_vals = self.bezier_y(t_vals)

            result[mask] = y_vals

        return result

    @property
    @override
    def annotations(self):
        return [
            ConstAnnotation('input', 'angular', 'start', self.a),
            ConstAnnotation('output', 'angular', 'end', self.b),
        ]

    def __repr__(self):
        return f'BezierMapper(a={self.a:.2g}, b={self.b:.2g})'

    def __str__(self):
        components = ['Bézier']
        if self.a != 0:
            components.append(rf'$a = {self.a:.2g}$')
        if self.b != 1:
            components.append(rf'$b = {self.b:.2g}$')
        return rf'{", ".join(components)}'


class FastBezierMapper(BezierMapper):
    x_lookup: Tensor
    y_lookup: Tensor

    def __init__(
        self,
        a: Annotated[float, [Ge(0), Lt(1)]],
        b: Annotated[float, [Gt(0), Le(1)]],
        *,
        start_slope: float = 1.0,  # Aligned with unmapped leadup
        end_slope: float = 0.0,  # Flat
        control_distance: float = 1 / sqrt(2),  # Relative to intersection point
        lookup_resolution: int = 1000,
    ):
        super().__init__(
            a,
            b,
            start_slope=start_slope,
            end_slope=end_slope,
            control_distance=control_distance,
        )

        # Precompute lookup table
        t_vals = torch.linspace(0, 1, lookup_resolution, dtype=torch.float32)
        bezier_points = self.bezier_point(t_vals)

        # Register lookup tables as buffers
        self.register_buffer('x_lookup', bezier_points[:, 0].contiguous())  # x coordinates
        self.register_buffer('y_lookup', bezier_points[:, 1].contiguous())  # y coordinates

    def interpolate_1d(self, x_query: Tensor) -> Tensor:
        """1D linear interpolation using lookup table."""
        # Buffers are automatically on the correct device
        x_lookup = self.x_lookup
        y_lookup = self.y_lookup

        # Find insertion points for x_query in x_lookup
        indices = torch.searchsorted(x_lookup, x_query, right=False)

        # Clamp indices to valid range
        max_index = x_lookup.shape[0] - 1
        indices = torch.clamp(indices, 1, max_index)

        # Get surrounding points
        x0 = x_lookup[indices - 1]
        x1 = x_lookup[indices]
        y0 = y_lookup[indices - 1]
        y1 = y_lookup[indices]

        # Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        t = (x_query - x0) / (x1 - x0 + 1e-8)
        y_interp = y0 + t * (y1 - y0)

        return y_interp

    @override
    def forward(
        self,
        alignment: Tensor,  # [B]
    ) -> Tensor:  # [B]
        result = alignment.clone()
        mask = alignment > self.a

        if mask.any():
            x_vals = alignment[mask]

            # Use interpolation on lookup table instead of Newton's method
            y_vals = self.interpolate_1d(x_vals)

            result[mask] = y_vals

        return result
