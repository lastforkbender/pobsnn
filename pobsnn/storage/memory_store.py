from __future__ import annotations

from typing import Any

from .base_store import BaseStore
from .json_tools import to_jsonable


class MemoryStore(BaseStore):
    """Small deterministic store for tests and dry runs."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    def write_json(self, path: str, name: str, value: Any, *, overwrite: bool = True) -> None:
        bucket = self.data.setdefault(_clean_path(path), {})
        if name in bucket and not overwrite:
            raise FileExistsError(f"{path}/{name}")
        bucket[name] = to_jsonable(value)

    def read_json(self, path: str, name: str) -> Any:
        return self.data[_clean_path(path)][name]


def _clean_path(path: str) -> str:
    return "/" + path.strip("/")
