from __future__ import annotations

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.evolution import trace_from_training_state
from pobsnn.storage import MemoryStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig


def test_evolution_trace_is_created_and_persisted_inside_pobsnn_region():
    x = np.linspace(0.0, 1.0, 24)[:, None]
    y = np.sin(2.0 * np.pi * x)
    store = MemoryStore()
    layer = BSplineLayer(1, 1, degree=3, num_basis=7, seed=12)
    trainer = PolicyGatedTrainer(
        layer,
        store=store,
        config=TrainerConfig(run_id="trace_test", epochs=3, snapshot_interval=3),
    )

    history = trainer.train(x, y)
    trace = store.read_json("/experiments/trace_test/evolution/traces", "epoch_000003")

    assert trace["trace_id"] == "trace_test:epoch:000003"
    assert trace["trace_type"] == "training_epoch"
    assert trace["run_id"] == "trace_test"
    assert trace["epoch"] == 3
    assert "policy-gated" in trace["tags"]
    assert "svd" in trace["payload"]
    assert trace["payload"]["policy_decisions"] == history[-1].policy_decisions


def test_trace_from_training_state_keeps_ranking_reasoning_out_of_storage():
    x = np.linspace(0.0, 1.0, 16)[:, None]
    y = x * 0.5
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=2, num_basis=5, seed=7),
        config=TrainerConfig(run_id="boundary", epochs=1),
    )
    state = trainer.train(x, y)[0]

    trace = trace_from_training_state("boundary", state).to_dict()

    assert "rank" not in trace  # no storage-side ranking instruction
    assert "reasoning_boundary" not in trace  # manifest owns boundary text
    assert isinstance(trace["score"], float)
    assert isinstance(trace["compression_pressure"], float)
