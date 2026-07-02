from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class ProposalSeverity(str, Enum):
    INFO = "info"
    WATCH = "watch"
    CAUTION = "caution"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PolicyProposal:
    """A controller recommendation.

    Phase 1 proposals are observation-only. They do not mutate the engine.
    """

    controller_id: str
    severity: ProposalSeverity
    title: str
    rationale: str
    recommended_action: str
    may_mutate_engine: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data
