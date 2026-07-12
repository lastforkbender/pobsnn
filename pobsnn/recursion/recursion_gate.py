from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

from pobsnn.node_properties import NodePropertyState, NodeProperty
from pobsnn.recursion.compression_state import CompressionState


class RecursionDecision(str, Enum):
    HOLD = "hold"
    WATCH = "watch"
    SPAWN_OBSERVER = "spawn_observer"


@dataclass(frozen=True)
class RecursionGateResult:
    decision: RecursionDecision
    pressure: float
    score_delta: float
    target_property: str
    rationale: str

    def to_dict(self) -> dict[str, float | str]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


def evaluate_recursion_gate(
    compression: CompressionState,
    node_states: list[NodePropertyState],
    *,
    score_delta: float,
    spawn_pressure: float = 0.45,
) -> RecursionGateResult:
    counts = {p: 0 for p in NodeProperty}
    for s in node_states:
        counts[s.property] += 1

    if compression.pressure >= spawn_pressure and (
        counts[NodeProperty.REDUNDANT] > 0 or counts[NodeProperty.VOLATILE] > 0
    ):
        target = "redundant" if counts[NodeProperty.REDUNDANT] else "volatile"
        return RecursionGateResult(
            RecursionDecision.SPAWN_OBSERVER,
            compression.pressure,
            float(score_delta),
            target,
            "compression pressure and typed node state agree; spawn observer only",
        )

    if compression.pressure >= 0.25 or counts[NodeProperty.UNDERUTILIZED] > 0:
        return RecursionGateResult(
            RecursionDecision.WATCH,
            compression.pressure,
            float(score_delta),
            "underutilized" if counts[NodeProperty.UNDERUTILIZED] else "compression",
            "signal present but not enough agreement for spawning",
        )

    return RecursionGateResult(
        RecursionDecision.HOLD,
        compression.pressure,
        float(score_delta),
        "none",
        "no recursion signal",
    )
