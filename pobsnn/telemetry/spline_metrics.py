from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from pobsnn.core.bspline_layer import BSplineLayer


@dataclass(frozen=True)
class SplineHealth:
    active_basis_ratio: float
    coefficient_l2: float
    coefficient_max_abs: float
    bias_l2: float
    knot_min_gap: float
    knot_max_gap: float
    output_mean: float
    output_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def spline_health(layer: BSplineLayer, x: np.ndarray) -> SplineHealth:
    """Return CPU telemetry for a B-spline layer."""
    x_arr = np.asarray(x, dtype=np.float64)
    y = layer(x_arr)
    util = layer.basis_utilization(x_arr)
    knots = layer.basis.knots
    positive_gaps = np.diff(knots)
    positive_gaps = positive_gaps[positive_gaps > 0.0]

    return SplineHealth(
        active_basis_ratio=float(np.mean(util)),
        coefficient_l2=float(np.linalg.norm(layer.coefficients)),
        coefficient_max_abs=float(np.max(np.abs(layer.coefficients))),
        bias_l2=float(np.linalg.norm(layer.bias)),
        knot_min_gap=float(np.min(positive_gaps)) if positive_gaps.size else 0.0,
        knot_max_gap=float(np.max(positive_gaps)) if positive_gaps.size else 0.0,
        output_mean=float(np.mean(y)),
        output_std=float(np.std(y)),
    )
