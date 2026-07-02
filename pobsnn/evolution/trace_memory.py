from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace_score import TraceScore


@dataclass
class TraceMemory:
    """In-process POBSNN trace score memory.

    This is not a storage backend and not a TDS feature. It is the small current
    run memory used by controller advisors.
    """

    scores: list[TraceScore] = field(default_factory=list)

    def add(self, score: TraceScore) -> None:
        self.scores.append(score)

    def top(self, n: int = 5) -> list[TraceScore]:
        return sorted(self.scores, key=lambda item: item.rank_score, reverse=True)[:n]

    def action_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for score in self.scores:
            counts[score.recommended_action] = counts.get(score.recommended_action, 0) + 1
        return counts

    def average_rank_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.rank_score for s in self.scores) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.scores),
            "average_rank_score": self.average_rank_score(),
            "action_counts": self.action_counts(),
            "top": [s.to_dict() for s in self.top()],
        }
