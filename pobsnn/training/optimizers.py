from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pobsnn.core import BSplineLayer


@dataclass(frozen=True)
class GradientStep:
    learning_rate: float
    grad_coeff_l2: float
    grad_bias_l2: float

    def to_dict(self) -> dict[str, float]:
        return {
            "learning_rate": float(self.learning_rate),
            "grad_coeff_l2": float(self.grad_coeff_l2),
            "grad_bias_l2": float(self.grad_bias_l2),
        }


class FullBatchMSEOptimizer:
    """Explicit NumPy full-batch gradient descent for BSplineLayer."""

    def __init__(self, learning_rate: float = 0.05) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)

    def step(self, layer: BSplineLayer, x: np.ndarray, target: np.ndarray) -> GradientStep:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(target, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        if x_arr.shape[-1] != layer.in_features or y_arr.shape[-1] != layer.out_features:
            raise ValueError("x/y feature dimensions do not match layer")

        basis = layer.basis(x_arr)
        pred = np.einsum("nib,oib->no", basis, layer.coefficients) + layer.bias
        err = pred - y_arr
        grad_pred = (2.0 / x_arr.shape[0]) * err
        grad_coeff = np.einsum("no,nib->oib", grad_pred, basis)
        grad_bias = np.sum(grad_pred, axis=0)
        layer.coefficients -= self.learning_rate * grad_coeff
        layer.bias -= self.learning_rate * grad_bias
        return GradientStep(
            learning_rate=self.learning_rate,
            grad_coeff_l2=float(np.linalg.norm(grad_coeff)),
            grad_bias_l2=float(np.linalg.norm(grad_bias)),
        )
