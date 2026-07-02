from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceScore:
    """POBSNN-owned score for an EvolutionTrace.

    Ranking is reasoning/advisory logic inside POBSNN. Storage backends receive
    only this serialized result; they do not compute, interpret, or update it.
    """

    trace_id: str
    rank_score: float
    quality_score: float
    stability_score: float
    compression_score: float
    novelty_score: float
    recommended_action: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
