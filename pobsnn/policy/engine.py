from __future__ import annotations

from dataclasses import dataclass, field

from .proposal import PolicyProposal, ProposalSeverity


@dataclass
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


@dataclass
class PolicyEngine:
    """Phase 1 policy gate.

    It records recommendations but blocks mutation. This preserves a clean line
    between observing a change and making a change.
    """

    allow_mutation: bool = False
    decisions: list[PolicyDecision] = field(default_factory=list)

    def review(self, proposals: list[PolicyProposal]) -> list[PolicyDecision]:
        reviewed: list[PolicyDecision] = []
        for proposal in proposals:
            if proposal.may_mutate_engine and not self.allow_mutation:
                decision = PolicyDecision(proposal, "defer", "Phase 1 is observation-only.")
            elif proposal.severity == ProposalSeverity.CRITICAL:
                decision = PolicyDecision(proposal, "accept_alert", "Critical observation accepted as alert.")
            else:
                decision = PolicyDecision(proposal, "record", "Observation recorded.")
            reviewed.append(decision)
        self.decisions.extend(reviewed)
        return reviewed
