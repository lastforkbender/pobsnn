from __future__ import annotations

from typing import Any

import numpy as np

from pobsnn.core import BSplineLayer


def layer_snapshot(layer: BSplineLayer) -> dict[str, Any]:
    return {
        "type": "BSplineLayer",
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "degree": layer.basis.degree,
        "num_basis": layer.basis.num_basis,
        "domain_min": layer.basis.config.domain_min,
        "domain_max": layer.basis.config.domain_max,
        "knots": layer.basis.knots.tolist(),
        "coefficients": layer.coefficients.tolist(),
        "bias": layer.bias.tolist(),
    }


def layer_from_snapshot(snapshot: dict[str, Any]) -> BSplineLayer:
    layer = BSplineLayer(
        int(snapshot["in_features"]),
        int(snapshot["out_features"]),
        degree=int(snapshot["degree"]),
        num_basis=int(snapshot["num_basis"]),
        domain_min=float(snapshot["domain_min"]),
        domain_max=float(snapshot["domain_max"]),
    )
    layer.coefficients = np.asarray(snapshot["coefficients"], dtype=np.float64)
    layer.bias = np.asarray(snapshot["bias"], dtype=np.float64)
    return layer
