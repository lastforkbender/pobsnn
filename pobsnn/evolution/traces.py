from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvolutionTrace:
    """POBSNN-owned structural and recursive evidence trace."""

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
    communication_episode_id: str = ""
    route_trace_count: int = 0
    interval_mean_distance: float = 0.0
    interval_max_distance: float = 0.0
    cognitive_step_length: float = 0.0
    max_recursive_depth: int = 0
    source_confidence: float = 0.0
    source_recursive_params: tuple[dict[str, Any], ...] = ()
    tags: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trace_from_training_state(run_id: str, state: Any) -> EvolutionTrace:
    """Create a deterministic evolution trace from a TrainingState-like object."""

    compression = state.compression or {}
    svd = state.svd or {}
    gate = state.recursion_gate or {}
    episode = state.communication_episode or {}
    interval = episode.get("interval_field", {}) if isinstance(episode, dict) else {}
    length_rows = episode.get("cognitive_length", []) if isinstance(episode, dict) else []
    recursive_rows = episode.get("recursive_identities", []) if isinstance(episode, dict) else []
    pressure = float(compression.get("pressure", 0.0))
    effective_rank = int(svd.get("effective_rank", 0))
    action = str(gate.get("action", gate.get("decision", "observe")))
    spawn_requested = state.spawn_proposal is not None

    tags = ["training", "cpu", "policy-gated", "structurally-policy-gated", "recursive-evidence"]
    if pressure > 0.66:
        tags.append("compression-high")
    elif pressure > 0.33:
        tags.append("compression-medium")
    else:
        tags.append("compression-low")
    if spawn_requested:
        tags.append("spawn-proposed")
    if state.route_traces:
        tags.append("communication-routed")

    total_step_length = sum(float(row.get("step_length", 0.0)) for row in length_rows)
    max_depth = max((int(row.get("depth", 0)) for row in recursive_rows), default=0)
    positions = interval.get("positions", []) if isinstance(interval, dict) else []
    source_confidence = 0.0
    if positions:
        source_confidence = sum(float(row.get("confidence", 0.0)) for row in positions) / len(positions)

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
        communication_episode_id=str(episode.get("episode_id", "")) if isinstance(episode, dict) else "",
        route_trace_count=len(state.route_traces),
        interval_mean_distance=float(interval.get("mean_pair_distance", 0.0)) if isinstance(interval, dict) else 0.0,
        interval_max_distance=float(interval.get("max_pair_distance", 0.0)) if isinstance(interval, dict) else 0.0,
        cognitive_step_length=float(total_step_length),
        max_recursive_depth=max_depth,
        source_confidence=float(source_confidence),
        source_recursive_params=tuple(dict(row) for row in recursive_rows),
        tags=tags,
        payload={
            "svd": state.svd,
            "compression": state.compression,
            "recursion_gate": state.recursion_gate,
            "controller_proposals": state.controller_proposals,
            "policy_decisions": state.policy_decisions,
            "spawn_proposal": state.spawn_proposal,
            "route_traces": state.route_traces,
            "communication_episode": state.communication_episode,
        },
    )
