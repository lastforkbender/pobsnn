from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from pobsnn.core.bspline_layer import BSplineLayer


@dataclass(frozen=True)
class SVDSummary:
    """Low-rank structure report for a spline layer.

    This is analysis only in Phase 1. It never mutates the layer.
    """

    full_rank: int
    effective_rank: int
    energy_90_rank: int
    energy_95_rank: int
    energy_99_rank: int
    spectral_entropy: float
    condition_number: float
    top_singular_value: float
    smallest_singular_value: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def coefficient_matrix(layer: BSplineLayer) -> np.ndarray:
    """Flatten coefficients as out_features x expanded_input matrix."""
    return np.asarray(layer.coefficients, dtype=np.float64).reshape(layer.out_features, -1)


def svd_summary(layer: BSplineLayer, tol: float = 1e-10) -> SVDSummary:
    matrix = coefficient_matrix(layer)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return SVDSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    energy = singular * singular
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        cumulative = np.zeros_like(energy)
        probabilities = np.zeros_like(energy)
    else:
        cumulative = np.cumsum(energy) / total_energy
        probabilities = energy / total_energy

    def rank_for_energy(target: float) -> int:
        if total_energy <= 0.0:
            return 0
        return int(np.searchsorted(cumulative, target, side="left") + 1)

    positive = singular[singular > tol]
    effective_rank = int(positive.size)
    entropy = 0.0
    probs = probabilities[probabilities > 0.0]
    if probs.size:
        entropy = float(-np.sum(probs * np.log(probs)))
    condition = float(positive[0] / positive[-1]) if positive.size > 1 else 1.0

    return SVDSummary(
        full_rank=int(singular.size),
        effective_rank=effective_rank,
        energy_90_rank=rank_for_energy(0.90),
        energy_95_rank=rank_for_energy(0.95),
        energy_99_rank=rank_for_energy(0.99),
        spectral_entropy=entropy,
        condition_number=condition,
        top_singular_value=float(singular[0]),
        smallest_singular_value=float(singular[-1]),
    )
