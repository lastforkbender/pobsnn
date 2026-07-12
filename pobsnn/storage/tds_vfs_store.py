from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pobsnn.identity import stable_digest

from .base_store import BaseStore, StoreOperationError
from .json_tools import to_jsonable


class TDSVFSStore(BaseStore):
    """POBSNN adapter for the public Staqtapp-TDS v3.5.2 API.

    The adapter uses result-first JSON operations, managed CSV transactions,
    scan/row-anchor evidence, validated persistence, and content-neutral Spiral
    ordering.  It never places TDS inside POBSNN reasoning.
    """

    def __init__(
        self,
        mount_dir: str | Path,
        root_name: str = "pobsnn_memory",
        *,
        resume: bool = True,
        materialize_csv_scan_evidence: bool = True,
    ) -> None:
        try:
            from staqtapp_tds import TDSFileSystem, TDSPersistence, __version__ as tds_version  # type: ignore
        except Exception as exc:  # pragma: no cover - environment-specific import
            raise ImportError(
                "TDSVFSStore requires Staqtapp-TDS v3.5.2 to be installed or on PYTHONPATH. "
                "POBSNN keeps TDS as a separate dependency."
            ) from exc
        if not str(tds_version).startswith("3.5.2"):
            raise RuntimeError(f"POBSNN v1.5 requires the TDS v3.5.2 API; found {tds_version}")

        self.mount_dir = Path(mount_dir)
        self.root_name = str(root_name).strip() or "pobsnn_memory"
        self.fs = TDSFileSystem(self.root_name)
        self.persistence = TDSPersistence(self.mount_dir)
        self.persistence.mount(self.fs)
        self.resume = bool(resume)
        self.materialize_csv_scan_evidence = bool(materialize_csv_scan_evidence)
        self._nodes: dict[str, Any] = {}
        self._loaded_paths: set[str] = set()
        self.tds_version = str(tds_version)

    @property
    def supports_csv_artifacts(self) -> bool:
        return True

    @property
    def supports_trace_ordering(self) -> bool:
        return True

    def _persisted_node_path(self, clean_path: str) -> Path:
        suffix = clean_path.strip("/").replace("/", "__")
        filename = self.root_name if not suffix else f"{self.root_name}__{suffix}"
        return self.mount_dir / f"{filename}.tds"

    def node(self, path: str) -> Any:
        clean = _clean_path(path)
        cached = self._nodes.get(clean)
        if cached is not None:
            return cached
        node = self.fs.makedirs(clean)
        persisted = self._persisted_node_path(clean)
        if self.resume and persisted.exists() and clean not in self._loaded_paths:
            self.persistence.load_node(persisted, into=node)
            self._loaded_paths.add(clean)
        self._nodes[clean] = node
        return node

    @staticmethod
    def _require_result(result: Any, operation: str) -> Any:
        if result is None or not bool(getattr(result, "ok", False)):
            code = getattr(result, "code", "unknown")
            message = getattr(result, "message", "no result returned")
            raise StoreOperationError(f"TDS {operation} failed: {code}: {message}")
        return result

    def write_json(self, path: str, name: str, value: Any, *, overwrite: bool = True) -> dict[str, Any]:
        result = self.node(path).write_json(name, to_jsonable(value), overwrite=overwrite, provenance="DERIVED")
        self._require_result(result, "write_json")
        return {
            "ok": True,
            "code": str(result.code),
            "message": result.message,
            "path": result.path,
            "name": result.name,
            "meta": to_jsonable(result.meta),
        }

    def read_json(self, path: str, name: str) -> Any:
        result = self.node(path).read_result(name)
        self._require_result(result, "read_result")
        return result.value

    def write_csv_artifact(
        self,
        path: str,
        csv_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        from staqtapp_tds.csv_layer import (  # type: ignore
            begin_csv_artifact_transaction,
            commit_csv_artifact_transaction,
            artifact_keys,
            export_original_csv,
            load_csv_manifest,
            materialize_csv_scan_artifacts,
            validate_csv_artifact_transaction,
            validate_csv_artifacts,
            validate_materialized_csv_scan_artifacts,
        )

        materialized = [{str(k): to_jsonable(v) for k, v in dict(row).items()} for row in rows]
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)
        raw = out.getvalue().encode("utf-8")
        node = self.node(path)

        manifest_key = artifact_keys(csv_id)["manifest"]
        manifest_result = node.read_result(manifest_key)
        if manifest_result.ok:
            load_csv_manifest(node, csv_id)
            existing = export_original_csv(node, csv_id).encode("utf-8")
            if existing == raw and not overwrite:
                report = validate_csv_artifacts(node, csv_id)
                if not report.ok:
                    raise StoreOperationError(f"existing TDS CSV artifact failed validation: {report.errors}")
                return {
                    "ok": True,
                    "status": "already_committed",
                    "csv_id": csv_id,
                    "row_count": len(materialized),
                    "validation": report.to_dict(),
                }
            if not overwrite:
                raise FileExistsError(f"managed CSV artifact {csv_id!r} already exists with different bytes")
        elif "MISSING" not in str(manifest_result.code).upper():
            raise StoreOperationError(
                f"TDS CSV manifest probe failed: {manifest_result.code}: {manifest_result.message}"
            )

        transaction_id = "tx-" + stable_digest("pobsnn-csv-transaction", path, csv_id, raw, length=20)
        staged = begin_csv_artifact_transaction(
            node,
            raw,
            source_name=f"{csv_id}.csv",
            csv_id=csv_id,
            transaction_id=transaction_id,
            overwrite=overwrite,
        )
        if not staged.ok:
            raise StoreOperationError(f"TDS CSV stage failed: {staged.errors}")
        checked = validate_csv_artifact_transaction(node, csv_id, transaction_id)
        if not checked.ok:
            raise StoreOperationError(f"TDS CSV transaction validation failed: {checked.errors}")
        committed = commit_csv_artifact_transaction(
            node,
            csv_id,
            transaction_id,
            overwrite=overwrite,
        )
        if not committed.ok:
            raise StoreOperationError(f"TDS CSV commit failed: {committed.errors}")
        validation = validate_csv_artifacts(node, csv_id)
        if not validation.ok:
            raise StoreOperationError(f"TDS CSV artifact validation failed: {validation.errors}")

        scan_report = None
        scan_validation = None
        if self.materialize_csv_scan_evidence:
            scan_report = materialize_csv_scan_artifacts(
                node,
                csv_id,
                include_row_anchors=True,
                overwrite=overwrite,
            )
            if not scan_report.ok:
                raise StoreOperationError(f"TDS CSV scan materialization failed: {scan_report.errors}")
            scan_validation = validate_materialized_csv_scan_artifacts(node, csv_id)
            if not scan_validation.ok:
                raise StoreOperationError(f"TDS CSV scan evidence validation failed: {scan_validation.errors}")

        return {
            "ok": True,
            "status": committed.status,
            "csv_id": csv_id,
            "row_count": len(materialized),
            "transaction": committed.to_dict(),
            "validation": validation.to_dict(),
            "scan_materialization": None if scan_report is None else scan_report.to_dict(),
            "scan_validation": None if scan_validation is None else scan_validation.to_dict(),
        }

    def rank_trace_evidence(
        self,
        trace_ids: Sequence[str],
        scores: Sequence[float],
        *,
        confidences: Sequence[float],
        depths: Sequence[int],
        ages_ns: Sequence[int],
        limit: int | None = None,
    ) -> dict[str, Any]:
        from staqtapp_tds.spiral import NativeSpiralRankEngine, SpiralRankConfig  # type: ignore

        # POBSNN interprets recursive depth contextually.  TDS records the depth
        # coordinate but does not apply a universal depth or age penalty here.
        engine = NativeSpiralRankEngine(
            SpiralRankConfig(depth_penalty=0.0, age_penalty=0.0, config_id="pobsnn-v1.5-source-order"),
            prefer_native=True,
        )
        run = engine.rank_run(
            trace_ids,
            scores,
            confidences=confidences,
            depths=depths,
            ages_ns=ages_ns,
            limit=limit,
        )
        return run.to_dict()

    def integrity_snapshot(self) -> dict[str, Any]:
        from staqtapp_tds import verify  # type: ignore

        report = verify(self.fs)
        data = report.to_dict() if hasattr(report, "to_dict") else {"repr": repr(report)}
        return {
            "backend": type(self).__name__,
            "status": "available",
            "tds_version": self.tds_version,
            "mount_dir": str(self.mount_dir),
            "loaded_path_count": len(self._loaded_paths),
            "health": to_jsonable(data),
        }

    def flush(self) -> dict[str, int]:
        return self.persistence.flush(self.fs)

    def close(self) -> None:
        self.persistence.unmount()


def _clean_path(path: str) -> str:
    return "/" + path.strip("/")
