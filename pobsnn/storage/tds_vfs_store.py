from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_store import BaseStore
from .json_tools import to_jsonable


class TDSVFSStore(BaseStore):
    """Adapter from POBSNN semantic memory to Staqtapp-TDS v2.6 VFS.

    This adapter consumes the published TDS API only. It does not modify,
    monkeypatch, subclass, or require changes to the Staqtapp-TDS library.
    """

    def __init__(self, mount_dir: str | Path, root_name: str = "pobsnn_memory") -> None:
        try:
            from staqtapp_tds import TDSFileSystem, TDSPersistence  # type: ignore
        except Exception as exc:  # pragma: no cover - exact environment-specific import
            raise ImportError(
                "TDSVFSStore requires staqtapp_tds to be installed or on PYTHONPATH. "
                "POBSNN keeps TDS as a separate dependency."
            ) from exc

        self.mount_dir = Path(mount_dir)
        self.fs = TDSFileSystem(root_name)
        self.persistence = TDSPersistence(self.mount_dir)
        self._nodes: dict[str, Any] = {}

    def node(self, path: str) -> Any:
        clean = _clean_path(path)
        cached = self._nodes.get(clean)
        if cached is not None:
            return cached
        node = self.fs.makedirs(clean)
        self._nodes[clean] = node
        return node

    def write_json(self, path: str, name: str, value: Any, *, overwrite: bool = True) -> None:
        self.node(path).write_json(name, to_jsonable(value), overwrite=overwrite)

    def read_json(self, path: str, name: str) -> Any:
        return self.node(path).read(name)

    def flush(self) -> None:
        self.persistence.flush(self.fs)


def _clean_path(path: str) -> str:
    return "/" + path.strip("/")
