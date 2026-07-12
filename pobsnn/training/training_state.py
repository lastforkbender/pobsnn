from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingState:
    epoch: int
    loss: float
    score: float
    score_delta: float
    svd: dict[str, Any]
    compression: dict[str, Any]
    node_states: list[dict[str, Any]]
    recursion_gate: dict[str, Any]
    controller_proposals: list[dict[str, Any]] = field(default_factory=list)
    policy_decisions: list[dict[str, Any]] = field(default_factory=list)
    spawn_proposal: dict[str, Any] | None = None
    gradient_step: dict[str, Any] | None = None
    route_traces: list[dict[str, Any]] = field(default_factory=list)
    communication_episode: dict[str, Any] | None = None
    evolution_trace: dict[str, Any] | None = None
    trace_score: dict[str, Any] | None = None
    tds_rank_run: dict[str, Any] | None = None
    strategy_advice: dict[str, Any] | None = None
    controller_update: dict[str, Any] | None = None
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
