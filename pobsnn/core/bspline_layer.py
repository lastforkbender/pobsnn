from __future__ import annotations

import numpy as np

from .spline_basis import BSplineBasis


class BSplineLayer:
    """Vectorized CPU-only B-spline layer.

    Input shape: ``(..., in_features)``
    Output shape: ``(..., out_features)``

    Each input feature is expanded by the same fixed spline basis. Outputs are
    learned linear combinations over all expanded basis values.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 3,
        num_basis: int = 12,
        domain_min: float = -1.0,
        domain_max: float = 1.0,
        init_scale: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.basis = BSplineBasis(degree, num_basis, domain_min, domain_max)
        rng = np.random.default_rng(seed)
        self.coefficients = rng.uniform(
            -init_scale, init_scale, size=(out_features, in_features, num_basis)
        )
        self.bias = np.zeros(out_features, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x_arr.shape[-1]}")
        B = self.basis(x_arr)  # (..., in_features, num_basis)
        return np.einsum("...ib,oib->...o", B, self.coefficients) + self.bias

    __call__ = forward

    def fit_coefficients_mse(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        learning_rate: float = 0.05,
    ) -> list[float]:
        """Manual full-batch gradient descent over coefficients and bias.

        This is intentionally explicit for Phase 0 learning and hand inspection.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[-1] != self.in_features or y_arr.shape[-1] != self.out_features:
            raise ValueError("x/y feature dimensions do not match this layer")

        losses: list[float] = []
        n_samples = x_arr.shape[0]
        if n_samples == 0:
            raise ValueError("x and y must contain at least one sample")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and y must contain the same number of samples")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("x and y must contain only finite values")
        B = self.basis(x_arr)  # precompute fixed basis for Phase 0
        for _ in range(epochs):
            pred = np.einsum("nib,oib->no", B, self.coefficients) + self.bias
            err = pred - y_arr
            loss = float(np.mean(err * err))
            losses.append(loss)

            grad_pred = (2.0 / (n_samples * self.out_features)) * err
            grad_coeff = np.einsum("no,nib->oib", grad_pred, B)
            grad_bias = np.sum(grad_pred, axis=0)
            candidate_coefficients = self.coefficients - learning_rate * grad_coeff
            candidate_bias = self.bias - learning_rate * grad_bias
            if not (np.all(np.isfinite(candidate_coefficients)) and np.all(np.isfinite(candidate_bias))):
                raise FloatingPointError("non-finite candidate coefficient update rejected")
            self.coefficients[...] = candidate_coefficients
            self.bias[...] = candidate_bias
        return losses

    def basis_utilization(self, x: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
        B = self.basis(np.asarray(x, dtype=np.float64))
        axes = tuple(range(B.ndim - 1))
        return (np.mean(np.abs(B), axis=axes) > threshold).astype(np.float64)
