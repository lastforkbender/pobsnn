from __future__ import annotations

from dataclasses import dataclass, asdict

from pobsnn.telemetry.svd_metrics import SVDSummary


@dataclass(frozen=True)
class CompressionState:
    pressure: float
    effective_rank_ratio: float
    redundancy_ratio: float
    rank_gap: int
    spectral_entropy: float
    condition_number: float
    rationale: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def compression_state_from_svd(svd: SVDSummary) -> CompressionState:
    if svd.full_rank <= 0:
        return CompressionState(0.0, 0.0, 0.0, 0, 0.0, 0.0, "empty SVD summary")

    eff_ratio = svd.effective_rank / svd.full_rank
    redundancy = max(0.0, 1.0 - (svd.energy_95_rank / svd.full_rank))
    rank_gap = max(0, svd.full_rank - svd.energy_95_rank)

    # Pressure combines low-rank redundancy and numerical conditioning.
    condition_term = min(1.0, max(0.0, (svd.condition_number - 10.0) / 90.0))
    pressure = float(min(1.0, 0.75 * redundancy + 0.25 * condition_term))

    if pressure >= 0.55:
        rationale = "high low-rank redundancy; compression observer eligible"
    elif pressure >= 0.25:
        rationale = "moderate low-rank structure; monitor before action"
    else:
        rationale = "low compression pressure"

    return CompressionState(
        pressure=pressure,
        effective_rank_ratio=float(eff_ratio),
        redundancy_ratio=float(redundancy),
        rank_gap=int(rank_gap),
        spectral_entropy=float(svd.spectral_entropy),
        condition_number=float(svd.condition_number),
        rationale=rationale,
    )
