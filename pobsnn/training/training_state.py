from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
