from __future__ import annotations

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.storage import MemoryStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig


def test_trainer_executes_complete_communication_episode() -> None:
    x = np.linspace(-1.0, 1.0, 48)[:, None]
    y = np.sin(np.pi * x)
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=3, num_basis=9, seed=201),
        config=TrainerConfig(run_id="episode", epochs=2, snapshot_interval=2),
    )
    state = trainer.train(x, y)[-1]
    episode = state.communication_episode
    assert episode is not None
    assert episode["episode_id"].startswith("EP-")
    assert episode["stride_id"].startswith("ST-")
    assert len(episode["recursive_identities"]) >= 6
    assert episode["interval_field"]["pair_count"] > 0
    assert episode["route_traces"]
    assert any(trace["responses"] for trace in episode["route_traces"])
    assert state.trace_score is not None
    assert "novelty_score" not in state.trace_score
    assert episode["maturity_boundary"]["novelty_score"] == "prohibited"


def test_trace_reasoning_runs_when_epoch_persistence_is_disabled() -> None:
    x = np.linspace(-1.0, 1.0, 32)[:, None]
    y = x * x
    store = MemoryStore()
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=2, num_basis=7, seed=202),
        store=store,
        config=TrainerConfig(
            run_id="no_epoch_persist",
            epochs=4,
            snapshot_interval=4,
            persist_every_epoch=False,
        ),
    )
    history = trainer.train(x, y)
    assert len(history) == 4
    assert len(trainer.trace_memory.scores) == 4
    assert all(state.trace_score is not None for state in history)
    assert "/experiments/no_epoch_persist/evolution/traces" not in store.data


def test_hot_histories_are_bounded() -> None:
    x = np.linspace(-1.0, 1.0, 24)[:, None]
    y = np.cos(np.pi * x)
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=2, num_basis=6, seed=203),
        config=TrainerConfig(
            run_id="bounded",
            epochs=9,
            snapshot_interval=9,
            history_limit=3,
            trace_memory_limit=4,
            router_history_limit=5,
        ),
    )
    history = trainer.train(x, y)
    assert len(history) == 3
    assert len(trainer.trace_memory.scores) == 4
    assert len(trainer.router.history) == 5


def test_memory_store_receives_residual_csv_families() -> None:
    x = np.linspace(-1.0, 1.0, 24)[:, None]
    y = x
    store = MemoryStore()
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=2, num_basis=6, seed=204),
        store=store,
        config=TrainerConfig(run_id="csv_residual", epochs=1, snapshot_interval=1),
    )
    trainer.train(x, y)
    path = "/experiments/csv_residual/evidence/stride_000001"
    keys = set(store.data[path])
    assert any("controller_positions" in key and key.endswith("raw.csv") for key in keys)
    assert any("vote_intervals" in key and key.endswith("raw.csv") for key in keys)
    assert any("cognitive_length" in key and key.endswith("raw.csv") for key in keys)
    assert any("route_responses" in key and key.endswith("raw.csv") for key in keys)
