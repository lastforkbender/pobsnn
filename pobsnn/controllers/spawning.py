from __future__ import annotations

from dataclasses import dataclass, asdict

from pobsnn.controllers.base import ControllerContext, MetaController
from pobsnn.node_properties import NodePropertyState
from pobsnn.policy import PolicyProposal, ProposalSeverity
from pobsnn.recursion import CompressionState, RecursionGateResult, RecursionDecision


@dataclass(frozen=True)
class SpawnProposal:
    parent_controller_id: str
    child_controller_id: str
    scope: str
    reason: str
    expires_after_observations: int = 5
    observation_only: bool = True

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


class SpawnedObserver(MetaController):
    """Observation-only child controller created by the spawn arbiter."""

    def __init__(self, proposal: SpawnProposal) -> None:
        self.controller_id = proposal.child_controller_id
        self.name = f"spawned-{proposal.scope}-observer"
        self.proposal = proposal
        self.remaining_observations = proposal.expires_after_observations

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        self.remaining_observations -= 1
        return [
            PolicyProposal(
                controller_id=self.controller_id,
                severity=ProposalSeverity.INFO,
                title=f"Spawned observer active: {self.proposal.scope}",
                rationale=self.proposal.reason,
                recommended_action="Continue observation; no engine mutation permitted.",
                may_mutate_engine=False,
            )
        ]

    @property
    def expired(self) -> bool:
        return self.remaining_observations <= 0


class SpawnArbiter(MetaController):
    controller_id = "MC-5"
    name = "spawn-arbiter"

    def propose_spawn(
        self,
        gate: RecursionGateResult,
        compression: CompressionState,
        node_states: list[NodePropertyState],
    ) -> SpawnProposal | None:
        if gate.decision is not RecursionDecision.SPAWN_OBSERVER:
            return None
        matching = [s for s in node_states if s.property.value == gate.target_property]
        node_scope = ",".join(str(s.node_index) for s in matching[:8]) or "layer"
        child_id = f"MC-5.{gate.target_property}.{abs(hash((node_scope, round(compression.pressure, 3)))) % 10000:04d}"
        return SpawnProposal(
            parent_controller_id=self.controller_id,
            child_controller_id=child_id,
            scope=f"{gate.target_property}:nodes[{node_scope}]",
            reason=f"{gate.rationale}; pressure={compression.pressure:.3f}; rank_gap={compression.rank_gap}",
        )

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        return [
            PolicyProposal(
                controller_id=self.controller_id,
                severity=ProposalSeverity.INFO,
                title="Spawn arbiter online",
                rationale="Phase 2 allows observer spawning only through explicit recursion gate results.",
                recommended_action="Use propose_spawn(gate, compression, node_states) after telemetry evaluation.",
                may_mutate_engine=False,
            )
        ]
