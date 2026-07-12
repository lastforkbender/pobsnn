from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .proposal import PolicyProposal, ProposalSeverity


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    proposal: PolicyProposal
    decision: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal": self.proposal.to_dict(),
            "decision": self.decision,
            "reason": self.reason,
        }


class PolicyEngine:
    """Observation-oriented policy gate with bounded hot decision memory."""

    __slots__ = ("allow_mutation", "decisions")

    def __init__(self, allow_mutation: bool = False, *, max_decisions: int = 2048) -> None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        self.allow_mutation = bool(allow_mutation)
        self.decisions: deque[PolicyDecision] = deque(maxlen=max_decisions)

    def review(self, proposals: list[PolicyProposal]) -> list[PolicyDecision]:
        reviewed: list[PolicyDecision] = []
        for proposal in proposals:
            if proposal.may_mutate_engine and not self.allow_mutation:
                decision = PolicyDecision(proposal, "defer", "v1.5 structural mutation remains policy-blocked.")
            elif proposal.severity == ProposalSeverity.CRITICAL:
                decision = PolicyDecision(proposal, "accept_alert", "Critical observation accepted as alert.")
            else:
                decision = PolicyDecision(proposal, "record", "Observation recorded.")
            reviewed.append(decision)
        self.decisions.extend(reviewed)
        return reviewed
