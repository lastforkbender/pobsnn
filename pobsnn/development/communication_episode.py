from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pobsnn.routing import RouteTrace

from .cognitive_length import CognitiveLengthStep
from .interval_field import IntervalFieldSnapshot
from .recursive_identity import RecursiveIdentity


@dataclass(frozen=True, slots=True)
class CommunicationEpisode:
    """One bounded, source-preserving v1.5 process stride."""

    episode_id: str
    stride_id: str
    run_id: str
    epoch: int
    recursive_identities: tuple[RecursiveIdentity, ...]
    route_traces: tuple[RouteTrace, ...]
    interval_field: IntervalFieldSnapshot
    cognitive_length: tuple[CognitiveLengthStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "stride_id": self.stride_id,
            "run_id": self.run_id,
            "epoch": int(self.epoch),
            "recursive_identities": [item.to_dict() for item in self.recursive_identities],
            "route_traces": [item.to_dict() for item in self.route_traces],
            "interval_field": self.interval_field.to_dict(),
            "cognitive_length": [item.to_dict() for item in self.cognitive_length],
            "maturity_boundary": {
                "entropy_semantics": "reserved_not_implemented",
                "origamic_union": "reserved_not_implemented",
                "prospective_cognitive_arrival": "reserved_not_implemented",
                "novelty_score": "prohibited",
            },
        }
