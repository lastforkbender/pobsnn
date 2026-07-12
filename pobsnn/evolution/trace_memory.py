from __future__ import annotations

from collections import deque
from typing import Any

from .trace_score import TraceScore


class TraceMemory:
    """Bounded current-run POBSNN trace score memory."""

    __slots__ = ("scores",)

    def __init__(self, max_scores: int = 1024) -> None:
        if max_scores <= 0:
            raise ValueError("max_scores must be positive")
        self.scores: deque[TraceScore] = deque(maxlen=max_scores)

    def add(self, score: TraceScore) -> None:
        self.scores.append(score)

    def top(self, n: int = 5) -> list[TraceScore]:
        return sorted(self.scores, key=lambda item: (item.rank_score, item.trace_id), reverse=True)[:n]

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
            "capacity": self.scores.maxlen,
            "average_rank_score": self.average_rank_score(),
            "action_counts": self.action_counts(),
            "top": [s.to_dict() for s in self.top()],
        }
