from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ControllerUpdateDecision:
    accepted: bool
    action: str
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControllerUpdatePolicy:
    """Policy gate for controller self-improvement advice.

    In v1.3.0, the gate never applies code changes. It only accepts/rejects
    advisory strategy changes for reporting and future manual review.
    """

    def __init__(self, minimum_confidence: float = 0.20) -> None:
        self.minimum_confidence = float(minimum_confidence)

    def decide(self, advice: Any) -> ControllerUpdateDecision:
        confidence = float(getattr(advice, "confidence", 0.0))
        action = str(getattr(advice, "recommendation", "observe_more"))
        if action == "observe_more":
            return ControllerUpdateDecision(False, action, "insufficient_trace_evidence", confidence)
        if confidence < self.minimum_confidence:
            return ControllerUpdateDecision(False, action, "confidence_below_policy_threshold", confidence)
        return ControllerUpdateDecision(True, action, "accepted_as_advisory_only_no_mutation", confidence)
