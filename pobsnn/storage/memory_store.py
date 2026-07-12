from __future__ import annotations

import csv
import io
from typing import Any, Iterable, Mapping, Sequence

from pobsnn.identity import stable_digest

from .base_store import BaseStore
from .json_tools import to_jsonable


class MemoryStore(BaseStore):
    """Small deterministic store for tests and dry runs."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    @property
    def supports_csv_artifacts(self) -> bool:
        return True

    def write_json(self, path: str, name: str, value: Any, *, overwrite: bool = True) -> dict[str, Any]:
        bucket = self.data.setdefault(_clean_path(path), {})
        if name in bucket and not overwrite:
            raise FileExistsError(f"{path}/{name}")
        bucket[name] = to_jsonable(value)
        return {"ok": True, "status": "written", "path": _clean_path(path), "name": name}

    def read_json(self, path: str, name: str) -> Any:
        return self.data[_clean_path(path)][name]

    def write_csv_artifact(
        self,
        path: str,
        csv_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        materialized = [dict(row) for row in rows]
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
        text = out.getvalue()
        name = f"csv__{csv_id}__raw.csv"
        self.write_json(path, name, text, overwrite=overwrite)
        manifest = {
            "ok": True,
            "status": "memory_committed",
            "csv_id": csv_id,
            "row_count": len(materialized),
            "fieldnames": list(fieldnames),
            "source_fingerprint": stable_digest("memory-csv", text, length=64),
            "per_row_writes": False,
            "per_cell_writes": False,
        }
        self.write_json(path, f"csv__{csv_id}__pobsnn_manifest", manifest, overwrite=overwrite)
        return manifest

    def integrity_snapshot(self) -> dict[str, Any]:
        return {
            "backend": type(self).__name__,
            "status": "available",
            "directory_count": len(self.data),
            "entry_count": sum(len(v) for v in self.data.values()),
        }


def _clean_path(path: str) -> str:
    return "/" + path.strip("/")
