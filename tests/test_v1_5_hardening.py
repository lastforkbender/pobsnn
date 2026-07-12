from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pobsnn.core import BSplineLayer
from pobsnn.routing import make_message
from pobsnn.training import FullBatchMSEOptimizer, TrainerConfig


def test_multi_output_mse_gradient_matches_finite_difference() -> None:
    x = np.array([[-0.5, 0.2], [0.1, -0.3], [0.7, 0.8]], dtype=np.float64)
    y = np.array([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2], [0.9, -0.4, 0.1]], dtype=np.float64)
    layer = BSplineLayer(2, 3, degree=2, num_basis=5, seed=101)
    before = layer.coefficients.copy()
    learning_rate = 1e-4
    step = FullBatchMSEOptimizer(learning_rate).step(layer, x, y)
    assert step.accepted
    analytic = (before - layer.coefficients) / learning_rate

    probe = (1, 0, 2)
    epsilon = 1e-6
    layer.coefficients[...] = before
    layer.coefficients[probe] += epsilon
    loss_plus = float(np.mean((layer(x) - y) ** 2))
    layer.coefficients[...] = before
    layer.coefficients[probe] -= epsilon
    loss_minus = float(np.mean((layer(x) - y) ** 2))
    numeric = (loss_plus - loss_minus) / (2.0 * epsilon)

    assert analytic[probe] == pytest.approx(numeric, rel=2e-5, abs=2e-6)


def test_nonfinite_candidate_is_rejected_without_committing() -> None:
    x = np.array([[1.0], [0.5]], dtype=np.float64)
    y = np.array([[1e308], [-1e308]], dtype=np.float64)
    layer = BSplineLayer(1, 1, degree=1, num_basis=3, seed=102)
    coeff = layer.coefficients.copy()
    bias = layer.bias.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        step = FullBatchMSEOptimizer(1e308).step(layer, x, y)
    assert not step.accepted
    assert step.rejection_reason == "non_finite_candidate_update"
    assert np.array_equal(layer.coefficients, coeff)
    assert np.array_equal(layer.bias, bias)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": ""},
        {"epochs": 0},
        {"snapshot_interval": 0},
        {"learning_rate": 0.0},
        {"history_limit": 0},
    ],
)
def test_trainer_config_fails_fast(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainerConfig(**kwargs)


def test_route_payload_is_deeply_immutable_and_identity_includes_payload() -> None:
    payload = {"nested": {"values": [1, 2]}}
    first = make_message("PolicyReviewRequest", "test", epoch=1, payload=payload)
    payload["nested"]["values"].append(3)
    second = make_message("PolicyReviewRequest", "test", epoch=1, payload={"nested": {"values": [1, 2, 3]}})

    assert first.to_dict()["payload"] == {"nested": {"values": [1, 2]}}
    assert first.message_id != second.message_id
    with pytest.raises(TypeError):
        first.payload["new"] = 1  # type: ignore[index]


def test_spawn_identity_is_stable_across_python_hash_seeds(tmp_path: Path) -> None:
    script = tmp_path / "spawn_id.py"
    script.write_text(
        """
import numpy as np
from pobsnn.controllers.spawning import SpawnArbiter
from pobsnn.core import BSplineLayer
from pobsnn.node_properties import classify_nodes
from pobsnn.recursion import CompressionState, evaluate_recursion_gate

layer = BSplineLayer(1, 4, num_basis=6, seed=4)
x = np.linspace(-1, 1, 32)[:, None]
compression = CompressionState(0.8, 0.25, 0.75, 3, 0.2, 20.0, 'test')
states = classify_nodes(layer, x, score_delta=0.1, compression_pressure=0.8)
gate = evaluate_recursion_gate(compression, states, score_delta=0.1)
proposal = SpawnArbiter().propose_spawn(gate, compression, states)
print('none' if proposal is None else proposal.child_controller_id)
""".strip()
    )
    root = str(Path(__file__).resolve().parents[1])
    outputs = []
    for seed in ("1", "2", "3"):
        env = {"PYTHONPATH": root, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1
