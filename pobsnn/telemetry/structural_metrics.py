from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from pobsnn.core.bspline_layer import BSplineLayer


@dataclass(frozen=True)
class StructuralTelemetry:
    """Phase 1 geometry-focused telemetry for observation-only controllers."""

    mse_loss: float
    active_basis_ratio: float
    basis_entropy: float
    mean_basis_overlap: float
    coefficient_l2: float
    coefficient_drift_l2: float
    curvature_mean_abs: float
    curvature_max_abs: float
    output_mean: float
    output_std: float
    output_min: float
    output_max: float
    numerical_finite: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    p = values / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def structural_telemetry(
    layer: BSplineLayer,
    x: np.ndarray,
    target: np.ndarray | None = None,
    previous_coefficients: np.ndarray | None = None,
) -> StructuralTelemetry:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    y = layer(x_arr)
    B = layer.basis(x_arr)
    util = layer.basis_utilization(x_arr)
    mean_basis = np.mean(np.abs(B), axis=tuple(range(B.ndim - 1)))

    # Simple local curvature proxy across the sample dimension.
    if y.shape[0] >= 3:
        second = np.diff(y, n=2, axis=0)
        curvature_mean_abs = float(np.mean(np.abs(second)))
        curvature_max_abs = float(np.max(np.abs(second)))
    else:
        curvature_mean_abs = 0.0
        curvature_max_abs = 0.0

    if target is None:
        mse = float("nan")
    else:
        target_arr = np.asarray(target, dtype=np.float64)
        if target_arr.ndim == 1:
            target_arr = target_arr[:, None]
        err = y - target_arr
        mse = float(np.mean(err * err))

    if previous_coefficients is None:
        drift = 0.0
    else:
        drift = float(np.linalg.norm(layer.coefficients - previous_coefficients))

    return StructuralTelemetry(
        mse_loss=mse,
        active_basis_ratio=float(np.mean(util)),
        basis_entropy=_entropy(mean_basis),
        mean_basis_overlap=float(np.mean(np.sum(B > 1e-8, axis=-1))),
        coefficient_l2=float(np.linalg.norm(layer.coefficients)),
        coefficient_drift_l2=drift,
        curvature_mean_abs=curvature_mean_abs,
        curvature_max_abs=curvature_max_abs,
        output_mean=float(np.mean(y)),
        output_std=float(np.std(y)),
        output_min=float(np.min(y)),
        output_max=float(np.max(y)),
        numerical_finite=bool(np.all(np.isfinite(y)) and np.all(np.isfinite(layer.coefficients))),
    )
