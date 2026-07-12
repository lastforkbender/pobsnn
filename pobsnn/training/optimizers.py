from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pobsnn.core import BSplineLayer


@dataclass(frozen=True, slots=True)
class GradientStep:
    learning_rate: float
    grad_coeff_l2: float
    grad_bias_l2: float
    accepted: bool = True
    rejection_reason: str = ""
    candidate_coeff_l2: float = 0.0

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "learning_rate": float(self.learning_rate),
            "grad_coeff_l2": float(self.grad_coeff_l2),
            "grad_bias_l2": float(self.grad_bias_l2),
            "accepted": bool(self.accepted),
            "rejection_reason": self.rejection_reason,
            "candidate_coeff_l2": float(self.candidate_coeff_l2),
        }


class FullBatchMSEOptimizer:
    """Transactional NumPy full-batch gradient descent for ``BSplineLayer``."""

    def __init__(self, learning_rate: float = 0.05) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.learning_rate = float(learning_rate)

    def step(self, layer: BSplineLayer, x: np.ndarray, target: np.ndarray) -> GradientStep:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(target, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError("x and target must be one- or two-dimensional arrays")
        if x_arr.shape[0] == 0 or y_arr.shape[0] == 0:
            raise ValueError("x and target must contain at least one sample")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and target must contain the same number of samples")
        if x_arr.shape[-1] != layer.in_features or y_arr.shape[-1] != layer.out_features:
            raise ValueError("x/y feature dimensions do not match layer")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("x and target must contain only finite values")

        basis = layer.basis(x_arr)
        pred = np.einsum("nib,oib->no", basis, layer.coefficients) + layer.bias
        err = pred - y_arr

        # mse_loss() is the mean over every output element.  Its exact gradient
        # therefore divides by sample_count * out_features.
        denominator = x_arr.shape[0] * layer.out_features
        grad_pred = (2.0 / denominator) * err
        grad_coeff = np.einsum("no,nib->oib", grad_pred, basis)
        grad_bias = np.sum(grad_pred, axis=0)

        candidate_coefficients = layer.coefficients - self.learning_rate * grad_coeff
        candidate_bias = layer.bias - self.learning_rate * grad_bias
        grad_coeff_l2 = float(np.linalg.norm(grad_coeff))
        grad_bias_l2 = float(np.linalg.norm(grad_bias))
        candidate_coeff_l2 = float(np.linalg.norm(candidate_coefficients))

        if not (
            np.all(np.isfinite(grad_coeff))
            and np.all(np.isfinite(grad_bias))
            and np.all(np.isfinite(candidate_coefficients))
            and np.all(np.isfinite(candidate_bias))
        ):
            return GradientStep(
                learning_rate=self.learning_rate,
                grad_coeff_l2=grad_coeff_l2,
                grad_bias_l2=grad_bias_l2,
                accepted=False,
                rejection_reason="non_finite_candidate_update",
                candidate_coeff_l2=candidate_coeff_l2,
            )

        # Commit only after complete finite validation.
        layer.coefficients[...] = candidate_coefficients
        layer.bias[...] = candidate_bias
        return GradientStep(
            learning_rate=self.learning_rate,
            grad_coeff_l2=grad_coeff_l2,
            grad_bias_l2=grad_bias_l2,
            accepted=True,
            candidate_coeff_l2=candidate_coeff_l2,
        )
