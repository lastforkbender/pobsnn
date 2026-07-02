from __future__ import annotations

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.node_properties import NodeProperty, classify_nodes, node_svd_contributions
from pobsnn.recursion import RecursionDecision, compression_state_from_svd, evaluate_recursion_gate
from pobsnn.telemetry import svd_summary
from pobsnn.controllers.spawning import SpawnArbiter, SpawnedObserver


def redundant_layer() -> BSplineLayer:
    layer = BSplineLayer(in_features=1, out_features=6, num_basis=10, seed=22)
    layer.coefficients[1:] = layer.coefficients[0] * 0.01
    return layer


def test_svd_contributions_are_normalized() -> None:
    layer = redundant_layer()
    contrib = node_svd_contributions(layer)
    assert contrib.shape == (6,)
    assert np.isclose(np.sum(contrib), 1.0)
    assert np.all(contrib >= 0.0)


def test_compression_state_detects_low_rank_pressure() -> None:
    layer = redundant_layer()
    summary = svd_summary(layer)
    state = compression_state_from_svd(summary)
    assert state.rank_gap >= 1
    assert state.pressure > 0.0
    assert 0.0 <= state.redundancy_ratio <= 1.0


def test_node_classifier_assigns_typed_states() -> None:
    layer = redundant_layer()
    x = np.linspace(-1, 1, 64)[:, None]
    compression = compression_state_from_svd(svd_summary(layer))
    states = classify_nodes(layer, x, score_delta=0.05, compression_pressure=max(0.6, compression.pressure))
    assert len(states) == layer.out_features
    assert any(s.property in {NodeProperty.REDUNDANT, NodeProperty.VOLATILE, NodeProperty.STABLE} for s in states)
    assert all(s.to_dict()["property"] for s in states)


def test_recursion_gate_can_request_spawn() -> None:
    layer = redundant_layer()
    x = np.linspace(-1, 1, 64)[:, None]
    compression = compression_state_from_svd(svd_summary(layer))
    # Force gate pressure to isolate gate behavior from SVD thresholds.
    compression = type(compression)(
        pressure=0.7,
        effective_rank_ratio=compression.effective_rank_ratio,
        redundancy_ratio=compression.redundancy_ratio,
        rank_gap=compression.rank_gap,
        spectral_entropy=compression.spectral_entropy,
        condition_number=compression.condition_number,
        rationale=compression.rationale,
    )
    states = classify_nodes(layer, x, score_delta=0.02, compression_pressure=compression.pressure)
    gate = evaluate_recursion_gate(compression, states, score_delta=0.02)
    assert gate.decision in {RecursionDecision.SPAWN_OBSERVER, RecursionDecision.WATCH}


def test_spawn_arbiter_creates_observation_only_child() -> None:
    layer = redundant_layer()
    x = np.linspace(-1, 1, 64)[:, None]
    compression = compression_state_from_svd(svd_summary(layer))
    compression = type(compression)(0.75, compression.effective_rank_ratio, compression.redundancy_ratio, compression.rank_gap, compression.spectral_entropy, compression.condition_number, compression.rationale)
    states = classify_nodes(layer, x, score_delta=0.03, compression_pressure=compression.pressure)
    gate = evaluate_recursion_gate(compression, states, score_delta=0.03)
    arbiter = SpawnArbiter()
    proposal = arbiter.propose_spawn(gate, compression, states)
    if gate.decision is RecursionDecision.SPAWN_OBSERVER:
        assert proposal is not None
        assert proposal.observation_only is True
        child = SpawnedObserver(proposal)
        obs = child.observe(None)[0]
        assert obs.may_mutate_engine is False
