from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvolutionTrace:
    """POBSNN-owned structural learning trace.

    This object belongs to the POBSNN reasoning/adaptation region. Storage
    backends, including TDS VFS, receive only the serialized record and do not
    interpret, rank, aggregate, or reason over it.
    """

    trace_id: str
    run_id: str
    epoch: int
    trace_type: str
    score: float
    loss: float
    score_delta: float
    compression_pressure: float
    effective_rank: int
    recursion_gate_action: str
    controller_count: int
    policy_decision_count: int
    spawn_requested: bool
    tags: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trace_from_training_state(run_id: str, state: Any) -> EvolutionTrace:
    """Create a deterministic evolution trace from a TrainingState-like object."""

    compression = state.compression or {}
    svd = state.svd or {}
    gate = state.recursion_gate or {}
    pressure = float(compression.get("pressure", 0.0))
    effective_rank = int(svd.get("effective_rank", 0))
    action = str(gate.get("action", "observe"))
    spawn_requested = state.spawn_proposal is not None

    tags = ["training", "cpu", "policy-gated"]
    if pressure > 0.66:
        tags.append("compression-high")
    elif pressure > 0.33:
        tags.append("compression-medium")
    else:
        tags.append("compression-low")
    if spawn_requested:
        tags.append("spawn-proposed")

    return EvolutionTrace(
        trace_id=f"{run_id}:epoch:{state.epoch:06d}",
        run_id=run_id,
        epoch=state.epoch,
        trace_type="training_epoch",
        score=float(state.score),
        loss=float(state.loss),
        score_delta=float(state.score_delta),
        compression_pressure=pressure,
        effective_rank=effective_rank,
        recursion_gate_action=action,
        controller_count=len(state.controller_proposals),
        policy_decision_count=len(state.policy_decisions),
        spawn_requested=spawn_requested,
        tags=tags,
        payload={
            "svd": state.svd,
            "compression": state.compression,
            "recursion_gate": state.recursion_gate,
            "controller_proposals": state.controller_proposals,
            "policy_decisions": state.policy_decisions,
            "spawn_proposal": state.spawn_proposal,
        },
    )
