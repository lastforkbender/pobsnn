from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .message import RouteMessage


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Deterministic map from message types to semantically correct controllers."""

    routes: Mapping[str, tuple[str, ...]]
    default_route: tuple[str, ...] = ("MC-3",)

    def __post_init__(self) -> None:
        frozen = {str(k): tuple(str(v) for v in values) for k, values in self.routes.items()}
        object.__setattr__(self, "routes", MappingProxyType(frozen))

    @classmethod
    def default(cls) -> "RoutePolicy":
        return cls(
            routes={
                "CompressionAlert": ("MC-1", "MC-3", "MC-4", "MC-5"),
                "CurvatureDriftAlert": ("MC-2", "MC-3"),
                "NodeInstabilityAlert": ("MC-3", "MC-5"),
                "NumericalUpdateRejected": ("MC-0", "MC-3", "MC-5"),
                "SpawnCandidate": ("MC-3", "MC-4", "MC-5"),
                "MergeCandidate": ("MC-1", "MC-3", "MC-4"),
                "PolicyReviewRequest": ("MC-0", "MC-3", "MC-5"),
                "TraceRankingUpdate": ("MC-0", "MC-3", "MC-5"),
            }
        )

    def targets_for(self, message: RouteMessage) -> tuple[str, ...]:
        return self.routes.get(message.message_type, self.default_route)
