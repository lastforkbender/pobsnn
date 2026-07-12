from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pobsnn.identity import freeze_json, thaw_json

from .message import RouteMessage


@dataclass(frozen=True, slots=True)
class ControllerRouteResponse:
    controller_id: str
    accepted: bool
    confidence: float
    note: str
    proposal: Mapping[str, Any] | None = None
    recursive_id: str | None = None

    def __post_init__(self) -> None:
        if self.proposal is not None:
            object.__setattr__(self, "proposal", freeze_json(dict(self.proposal)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "accepted": bool(self.accepted),
            "confidence": float(self.confidence),
            "note": self.note,
            "proposal": None if self.proposal is None else thaw_json(self.proposal),
            "recursive_id": self.recursive_id,
        }


@dataclass(frozen=True, slots=True)
class RouteTrace:
    """Complete auditable route episode."""

    trace_id: str
    message: RouteMessage
    targets: tuple[str, ...]
    responses: tuple[ControllerRouteResponse, ...] = ()
    aggregate: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "responses", tuple(self.responses))
        object.__setattr__(self, "aggregate", freeze_json(dict(self.aggregate or {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "message": self.message.to_dict(),
            "targets": list(self.targets),
            "responses": [r.to_dict() for r in self.responses],
            "aggregate": thaw_json(self.aggregate),
        }
