from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pobsnn.identity import freeze_json, stable_digest, thaw_json


@dataclass(frozen=True, slots=True)
class RouteMessage:
    """Deeply immutable diagnostic message routed between controllers."""

    message_id: str
    message_type: str
    source: str
    severity: float = 0.0
    node_id: int | None = None
    epoch: int | None = None
    event_index: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_json(dict(self.payload)))
        object.__setattr__(self, "tags", tuple(str(v) for v in self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type,
            "source": self.source,
            "severity": float(self.severity),
            "node_id": self.node_id,
            "epoch": self.epoch,
            "event_index": int(self.event_index),
            "payload": thaw_json(self.payload),
            "tags": list(self.tags),
        }


def make_message(
    message_type: str,
    source: str,
    *,
    severity: float = 0.0,
    node_id: int | None = None,
    epoch: int | None = None,
    event_index: int = 0,
    payload: Mapping[str, Any] | None = None,
    tags: tuple[str, ...] = (),
) -> RouteMessage:
    safe_type = message_type.strip() or "GenericEvent"
    safe_source = source.strip() or "unknown"
    safe_payload = dict(payload or {})
    safe_tags = tuple(str(v) for v in tags)
    digest = stable_digest(
        "route-message-v1.5",
        safe_source,
        safe_type,
        epoch,
        node_id,
        int(event_index),
        safe_payload,
        safe_tags,
        length=32,
    )
    return RouteMessage(
        message_id=f"msg-{digest}",
        message_type=safe_type,
        source=safe_source,
        severity=float(max(0.0, min(1.0, severity))),
        node_id=node_id,
        epoch=epoch,
        event_index=int(event_index),
        payload=safe_payload,
        tags=safe_tags,
    )
