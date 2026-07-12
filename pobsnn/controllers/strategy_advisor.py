from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .controller_profile import ControllerProfile


@dataclass(frozen=True)
class StrategyAdvice:
    controller_id: str
    recommendation: str
    confidence: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StrategyAdvisor:
    """Converts ranked trace evidence into controller strategy advice."""

    def advise(self, profile: ControllerProfile) -> StrategyAdvice:
        action = profile.dominant_action()
        if profile.observed_scores == 0:
            return StrategyAdvice(profile.controller_id, "observe_more", 0.0, "no_ranked_trace_evidence")
        confidence = min(1.0, profile.observed_scores / 10.0) * max(0.0, profile.average_rank_score)
        rationale = f"dominant_trace_recommendation={action}; observed_scores={profile.observed_scores}"
        return StrategyAdvice(profile.controller_id, action, float(confidence), rationale)
