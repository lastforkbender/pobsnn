from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStore(ABC):
    """Persistence boundary for POBSNN semantic memory.

    POBSNN computes. Storage backends persist snapshots, telemetry, controller
    reports, SVD summaries, policy decisions, and training history.
    """

    @abstractmethod
    def write_json(self, path: str, name: str, value: Any, *, overwrite: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, path: str, name: str) -> Any:
        raise NotImplementedError

    def flush(self) -> None:
        """Persist pending state. In-memory stores may no-op."""
        return None

    def close(self) -> None:
        self.flush()
