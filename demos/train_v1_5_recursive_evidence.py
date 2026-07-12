from __future__ import annotations

import json

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.storage import MemoryStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig


x = np.linspace(-1.0, 1.0, 64)[:, None]
y = np.sin(np.pi * x)

store = MemoryStore()
trainer = PolicyGatedTrainer(
    BSplineLayer(1, 1, degree=3, num_basis=10, seed=15),
    store=store,
    config=TrainerConfig(
        run_id="v15_recursive_evidence_demo",
        epochs=6,
        learning_rate=0.06,
        snapshot_interval=3,
    ),
)

state = trainer.train(x, y)[-1]
episode = state.communication_episode or {}

print(f"final loss: {state.loss:.8f}")
print(f"episode: {episode.get('episode_id')}")
print(f"route traces: {len(episode.get('route_traces', []))}")
print(f"controller intervals: {episode.get('interval_field', {}).get('pair_count', 0)}")
print("trace score:")
print(json.dumps(state.trace_score, indent=2))
