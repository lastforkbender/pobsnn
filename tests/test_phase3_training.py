from __future__ import annotations

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.storage import MemoryStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig


def test_policy_gated_training_reduces_loss_and_persists_semantics() -> None:
    x = np.linspace(-1.0, 1.0, 96)[:, None]
    y = np.sin(np.pi * x)
    store = MemoryStore()
    layer = BSplineLayer(1, 1, degree=3, num_basis=14, seed=3)
    trainer = PolicyGatedTrainer(
        layer,
        store=store,
        config=TrainerConfig(run_id="test_run", epochs=30, learning_rate=0.08, snapshot_interval=10),
    )

    history = trainer.train(x, y)

    assert len(history) == 30
    assert history[-1].loss < history[0].loss
    base = "/experiments/test_run"
    assert store.read_json(base, "manifest")["cpu_only"] is True
    assert "effective_rank" in store.read_json(f"{base}/telemetry/svd", "epoch_000030")
    assert isinstance(store.read_json(f"{base}/node_states", "epoch_000030"), list)
    assert store.read_json(f"{base}/network/snapshots", "epoch_000030")["type"] == "BSplineLayer"


def test_training_state_contains_recursion_and_policy_records() -> None:
    x = np.linspace(-1.0, 1.0, 64)[:, None]
    y = x * x
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 2, degree=2, num_basis=10, seed=4),
        config=TrainerConfig(run_id="wide", epochs=5, learning_rate=0.04),
    )
    history = trainer.train(x, np.hstack([y, -y]))
    last = history[-1].to_dict()
    assert "compression" in last
    assert "recursion_gate" in last
    assert "controller_proposals" in last
    assert "policy_decisions" in last
