from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence


class StoreOperationError(RuntimeError):
    """Raised when an external result-first storage operation reports failure."""


class BaseStore(ABC):
    """Persistence boundary for POBSNN evidence.

    Reasoning is completed before these methods are called.  Advanced methods
    preserve residual evidence and deterministically order caller-supplied trace
    metadata; they do not define POBSNN semantics.
    """

    @abstractmethod
    def write_json(self, path: str, name: str, value: Any, *, overwrite: bool = True) -> Any:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, path: str, name: str) -> Any:
        raise NotImplementedError

    @property
    def supports_csv_artifacts(self) -> bool:
        return False

    @property
    def supports_trace_ordering(self) -> bool:
        return False

    def write_csv_artifact(
        self,
        path: str,
        csv_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError("this store does not support managed CSV artifacts")

    def rank_trace_evidence(
        self,
        trace_ids: Sequence[str],
        scores: Sequence[float],
        *,
        confidences: Sequence[float],
        depths: Sequence[int],
        ages_ns: Sequence[int],
        limit: int | None = None,
    ) -> dict[str, Any] | None:
        return None

    def integrity_snapshot(self) -> dict[str, Any]:
        return {"backend": type(self).__name__, "status": "available"}

    def flush(self) -> Any:
        """Persist pending state. In-memory stores may no-op."""
        return None

    def close(self) -> None:
        self.flush()
