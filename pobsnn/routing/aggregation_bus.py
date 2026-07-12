from __future__ import annotations

from dataclasses import dataclass

from .message import RouteMessage
from .route_trace import ControllerRouteResponse, RouteTrace


@dataclass(slots=True)
class AggregationBus:
    """Combines routed controller responses into a policy-facing summary."""

    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")

    def aggregate(
        self,
        message: RouteMessage,
        targets: tuple[str, ...],
        responses: list[ControllerRouteResponse],
    ) -> RouteTrace:
        usable = [r for r in responses if r.accepted and r.confidence >= self.min_confidence]
        mean_confidence = 0.0 if not responses else sum(r.confidence for r in responses) / len(responses)
        accepted_fraction = 0.0 if not responses else len(usable) / len(responses)
        action = "review" if usable else "observe"
        if message.severity >= 0.8 and usable:
            action = "escalate_to_policy"
        aggregate = {
            "action": action,
            "accepted_response_count": len(usable),
            "response_count": len(responses),
            "accepted_fraction": float(accepted_fraction),
            "mean_confidence": float(mean_confidence),
            "message_type": message.message_type,
        }
        return RouteTrace(
            trace_id=f"route:{message.message_id.removeprefix('msg-')}",
            message=message,
            targets=targets,
            responses=tuple(responses),
            aggregate=aggregate,
        )
