from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def test_tds_vfs_store_writes_through_separate_tds_library(tmp_path: Path) -> None:
    if importlib.util.find_spec("staqtapp_tds") is None:
        pytest.skip("staqtapp_tds is not installed; adapter is optional and external")

    from pobsnn.storage import TDSVFSStore

    store = TDSVFSStore(tmp_path)
    store.write_json("/experiments/adapter_test/telemetry", "epoch_000001", {"loss": 0.25})
    assert store.read_json("/experiments/adapter_test/telemetry", "epoch_000001")["loss"] == 0.25
    store.flush()
    assert any(p.suffix == ".tds" for p in tmp_path.iterdir())
