from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BSplineBasisConfig:
    """Configuration for a fixed open-uniform B-spline basis."""

    degree: int = 3
    num_basis: int = 12
    domain_min: float = -1.0
    domain_max: float = 1.0

    def validate(self) -> None:
        if self.degree < 0:
            raise ValueError("degree must be >= 0")
        if self.num_basis <= self.degree:
            raise ValueError("num_basis must be greater than degree")
        if not self.domain_min < self.domain_max:
            raise ValueError("domain_min must be less than domain_max")


class BSplineBasis:
    """CPU-only Cox-de Boor B-spline basis evaluator.

    Phase 0 is intentionally NumPy-only: no PyTorch, no CUDA, no hidden GPU path.
    The knot vector is fixed so training remains focused on control coefficients.
    """

    def __init__(
        self,
        degree: int = 3,
        num_basis: int = 12,
        domain_min: float = -1.0,
        domain_max: float = 1.0,
        knots: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.config = BSplineBasisConfig(degree, num_basis, domain_min, domain_max)
        self.config.validate()
        self.dtype = np.dtype(dtype)

        if knots is None:
            knots = self._make_open_uniform_knots(
                degree=degree,
                num_basis=num_basis,
                domain_min=domain_min,
                domain_max=domain_max,
                dtype=self.dtype,
            )
        else:
            knots = np.asarray(knots, dtype=self.dtype).copy()
            expected = num_basis + degree + 1
            if knots.size != expected:
                raise ValueError(f"expected {expected} knots, got {knots.size}")
            if np.any(knots[1:] < knots[:-1]):
                raise ValueError("knots must be non-decreasing")

        self.knots = knots

    @staticmethod
    def _make_open_uniform_knots(
        degree: int,
        num_basis: int,
        domain_min: float,
        domain_max: float,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        interior_count = num_basis - degree - 1
        if interior_count > 0:
            interior = np.linspace(domain_min, domain_max, interior_count + 2, dtype=dtype)[1:-1]
            return np.concatenate((
                np.full(degree + 1, domain_min, dtype=dtype),
                interior,
                np.full(degree + 1, domain_max, dtype=dtype),
            ))
        return np.concatenate((
            np.full(degree + 1, domain_min, dtype=dtype),
            np.full(degree + 1, domain_max, dtype=dtype),
        ))

    @property
    def degree(self) -> int:
        return self.config.degree

    @property
    def num_basis(self) -> int:
        return self.config.num_basis

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate basis functions.

        Args:
            x: Scalar or NumPy array of any shape.

        Returns:
            Array with shape ``x.shape + (num_basis,)``.
        """
        x_arr = np.asarray(x, dtype=self.dtype)
        original_shape = x_arr.shape
        x_flat = np.clip(x_arr.reshape(-1), self.config.domain_min, self.config.domain_max)

        knots = self.knots
        n = self.num_basis
        B = np.zeros((x_flat.size, n), dtype=self.dtype)

        for i in range(n):
            left = knots[i]
            right = knots[i + 1]
            active = (x_flat >= left) & (x_flat < right)
            if i == n - 1:
                active |= np.isclose(x_flat, knots[-1])
            B[:, i] = active.astype(self.dtype)

        for k in range(1, self.degree + 1):
            next_B = np.zeros_like(B)
            for i in range(n):
                denom_left = knots[i + k] - knots[i]
                if denom_left != 0.0:
                    next_B[:, i] += ((x_flat - knots[i]) / denom_left) * B[:, i]

                if i + 1 < n:
                    denom_right = knots[i + k + 1] - knots[i + 1]
                    if denom_right != 0.0:
                        next_B[:, i] += ((knots[i + k + 1] - x_flat) / denom_right) * B[:, i + 1]
            B = next_B

        return B.reshape(*original_shape, n)

    __call__ = evaluate
