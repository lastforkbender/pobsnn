from __future__ import annotations

import numpy as np

from .spline_basis import BSplineBasis


class BSplineNeuron:
    """Single CPU-only spline neuron for one scalar input."""

    def __init__(
        self,
        degree: int = 3,
        num_basis: int = 12,
        domain_min: float = -1.0,
        domain_max: float = 1.0,
        init_scale: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.basis = BSplineBasis(degree, num_basis, domain_min, domain_max)
        rng = np.random.default_rng(seed)
        self.coefficients = rng.uniform(-init_scale, init_scale, size=num_basis)
        self.bias = 0.0

    def forward(self, x: np.ndarray | float) -> np.ndarray:
        B = self.basis(x)
        return np.tensordot(B, self.coefficients, axes=([-1], [0])) + self.bias

    __call__ = forward
