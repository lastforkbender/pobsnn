from __future__ import annotations

from collections import deque
from typing import Protocol

from .message import RouteMessage
from .route_policy import RoutePolicy
from .route_trace import RouteTrace


class RouteStore(Protocol):
    def write_json(self, path: str, name: str, value: object, *, overwrite: bool = True) -> object: ...


class ControllerRouter:
    """CPU-light deterministic controller-message router with bounded history."""

    __slots__ = ("policy", "history", "store", "store_path")

    def __init__(
        self,
        policy: RoutePolicy | None = None,
        *,
        max_history: int = 256,
        store: RouteStore | None = None,
        store_path: str = "/routing/traces",
    ) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.policy = policy or RoutePolicy.default()
        self.history: deque[RouteTrace] = deque(maxlen=max_history)
        self.store = store
        self.store_path = store_path

    def targets_for(self, message: RouteMessage) -> tuple[str, ...]:
        return self.policy.targets_for(message)

    def route(self, message: RouteMessage) -> RouteTrace:
        trace = RouteTrace(
            trace_id=f"route:{message.message_id.removeprefix('msg-')}",
            message=message,
            targets=self.targets_for(message),
        )
        return self.record(trace)

    def record(self, trace: RouteTrace) -> RouteTrace:
        self.history.append(trace)
        self._persist(trace)
        return trace

    def _persist(self, trace: RouteTrace) -> None:
        if self.store is not None:
            name = trace.trace_id.replace("/", "_").replace(":", "_")
            self.store.write_json(self.store_path, name, trace.to_dict(), overwrite=False)
