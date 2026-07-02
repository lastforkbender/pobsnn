from __future__ import annotations

import numpy as np

from pobsnn.core.bspline_layer import BSplineLayer
from pobsnn.node_properties.property_state import NodeProperty, NodePropertyState
from pobsnn.telemetry.svd_metrics import coefficient_matrix


def node_svd_contributions(layer: BSplineLayer) -> np.ndarray:
    """Return per-output-node contribution in a low-rank coefficient space.

    Uses leverage-like row energy from left singular vectors weighted by singular
    energy. Values are normalized to sum to 1 when possible.
    """
    matrix = coefficient_matrix(layer)
    if matrix.size == 0:
        return np.zeros(layer.out_features, dtype=np.float64)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    energy_by_node = np.sum((u * singular) ** 2, axis=1)
    total = float(np.sum(energy_by_node))
    if total <= 0.0:
        return np.zeros(layer.out_features, dtype=np.float64)
    return energy_by_node / total


def classify_nodes(
    layer: BSplineLayer,
    x: np.ndarray,
    *,
    score_delta: float = 0.0,
    compression_pressure: float = 0.0,
    utilization_floor: float = 0.18,
    contribution_floor: float | None = None,
    volatility_score_delta: float = 0.025,
) -> list[NodePropertyState]:
    """Assign observation-only property types to output spline nodes.

    The classifier combines basis utilization, coefficient magnitude, SVD row
    contribution, compression pressure, and external score movement. It does not
    mutate the layer and is deterministic for a fixed layer/input.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    outputs = np.asarray(layer(x_arr), dtype=np.float64)
    if outputs.ndim == 1:
        outputs = outputs[:, None]

    # Output variation is a practical utilization proxy per node.
    span = np.ptp(outputs, axis=0)
    max_span = float(np.max(span)) if span.size else 0.0
    utilization = span / max_span if max_span > 0.0 else np.zeros(layer.out_features)

    coeff_norms = np.linalg.norm(layer.coefficients.reshape(layer.out_features, -1), axis=1)
    max_norm = float(np.max(coeff_norms)) if coeff_norms.size else 0.0
    norm_ratio = coeff_norms / max_norm if max_norm > 0.0 else np.zeros_like(coeff_norms)

    contributions = node_svd_contributions(layer)
    if contribution_floor is None:
        contribution_floor = 0.5 / max(1, layer.out_features)

    states: list[NodePropertyState] = []
    for i in range(layer.out_features):
        u = float(utilization[i])
        n = float(norm_ratio[i])
        c = float(contributions[i])
        rationale = []

        if compression_pressure > 0.45 and c < contribution_floor and u < utilization_floor:
            prop = NodeProperty.REDUNDANT
            rationale.append("low SVD contribution under compression pressure")
        elif u < utilization_floor and n < 0.35:
            prop = NodeProperty.UNDERUTILIZED
            rationale.append("low activation span and low coefficient norm")
        elif abs(score_delta) >= volatility_score_delta and compression_pressure > 0.25:
            prop = NodeProperty.VOLATILE
            rationale.append("score changed while compression pressure is active")
        elif score_delta < -volatility_score_delta and c >= contribution_floor:
            prop = NodeProperty.EMERGENT
            rationale.append("improving score with unique SVD contribution")
        else:
            prop = NodeProperty.STABLE
            rationale.append("no strong compression or volatility signal")

        states.append(
            NodePropertyState(
                node_index=i,
                property=prop,
                utilization=u,
                coefficient_norm=float(coeff_norms[i]),
                svd_contribution=c,
                score_delta=float(score_delta),
                compression_pressure=float(compression_pressure),
                rationale="; ".join(rationale),
            )
        )
    return states
