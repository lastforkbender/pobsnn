from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _require_tds() -> None:
    if importlib.util.find_spec("staqtapp_tds") is None:
        pytest.skip("Staqtapp-TDS v3.5.2 is not installed or on PYTHONPATH")


def test_tds_v352_result_first_resume_csv_and_spiral(tmp_path: Path) -> None:
    _require_tds()
    from pobsnn.storage import TDSVFSStore

    store = TDSVFSStore(tmp_path, root_name="v15_test", resume=True)
    result = store.write_json("/experiments/run", "manifest", {"release": "1.5.0"}, overwrite=False)
    assert result["ok"]
    csv_report = store.write_csv_artifact(
        "/experiments/run/evidence",
        "controller_positions",
        [
            {"controller_id": "MC-0", "axis": "loss", "position": 0.2},
            {"controller_id": "MC-3", "axis": "stability", "position": 0.7},
        ],
        fieldnames=("controller_id", "axis", "position"),
    )
    assert csv_report["ok"]
    assert csv_report["validation"]["ok"]
    assert csv_report["scan_validation"]["ok"]

    rank = store.rank_trace_evidence(
        ["trace-a", "trace-b"],
        [0.7, 0.9],
        confidences=[0.8, 0.9],
        depths=[4, 1],
        ages_ns=[0, 0],
    )
    assert rank["records"][0]["trace_id"] == "trace-b"
    assert rank["records"][0]["config_id"] == "pobsnn-v1.5-source-order"
    store.flush()
    store.close()

    reopened = TDSVFSStore(tmp_path, root_name="v15_test", resume=True)
    assert reopened.read_json("/experiments/run", "manifest")["release"] == "1.5.0"
    reopened.close()
