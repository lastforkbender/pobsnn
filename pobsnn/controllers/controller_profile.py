from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pobsnn.evolution import TraceScore


@dataclass
class ControllerProfile:
    """Evidence profile for a controller or controller family.

    v1.3.0 uses this as advisory memory only. It does not rewrite controller
    code and does not bypass policy review.
    """

    controller_id: str
    observed_scores: int = 0
    average_rank_score: float = 0.0
    recommended_actions: dict[str, int] = field(default_factory=dict)

    def update(self, score: TraceScore) -> None:
        n = self.observed_scores
        self.average_rank_score = ((self.average_rank_score * n) + score.rank_score) / (n + 1)
        self.observed_scores = n + 1
        action = score.recommended_action
        self.recommended_actions[action] = self.recommended_actions.get(action, 0) + 1

    def dominant_action(self) -> str:
        if not self.recommended_actions:
            return "observe_more"
        return max(self.recommended_actions.items(), key=lambda item: item[1])[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"dominant_action": self.dominant_action()}
