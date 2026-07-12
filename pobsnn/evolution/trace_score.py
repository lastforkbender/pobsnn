from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceScore:
    """POBSNN-owned source-aware score for an evolution trace.

    v1.5 intentionally has no novelty score. Relational movement describes
    interval/cognitive-length evidence without declaring a new abstraction.
    """

    trace_id: str
    rank_score: float
    quality_score: float
    stability_score: float
    compression_score: float
    relational_movement_score: float
    source_confidence: float
    recursive_depth: int
    recommended_action: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "rank_score": float(self.rank_score),
            "quality_score": float(self.quality_score),
            "stability_score": float(self.stability_score),
            "compression_score": float(self.compression_score),
            "relational_movement_score": float(self.relational_movement_score),
            "source_confidence": float(self.source_confidence),
            "recursive_depth": int(self.recursive_depth),
            "recommended_action": self.recommended_action,
            "reasons": list(self.reasons),
        }
