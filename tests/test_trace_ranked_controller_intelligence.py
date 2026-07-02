from __future__ import annotations

import numpy as np

from pobsnn.controllers import ControllerProfile, StrategyAdvisor
from pobsnn.core import BSplineLayer
from pobsnn.evolution import TraceRanker, trace_from_training_state
from pobsnn.policy import ControllerUpdatePolicy
from pobsnn.storage import MemoryStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig


def test_trace_ranker_scores_training_traces_deterministically() -> None:
    x = np.linspace(0.0, 1.0, 24)[:, None]
    y = np.sin(2.0 * np.pi * x)
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=3, num_basis=7, seed=21),
        config=TrainerConfig(run_id="ranker", epochs=2, learning_rate=0.05),
    )
    state = trainer.train(x, y)[-1]
    trace = trace_from_training_state("ranker", state)
    score = TraceRanker().score(trace)

    assert score.trace_id == trace.trace_id
    assert 0.0 <= score.rank_score <= 1.0
    assert 0.0 <= score.quality_score <= 1.0
    assert score.recommended_action in {
        "prefer_observer_spawn_over_prune",
        "simulate_merge_or_freeze_before_action",
        "reduce_controller_aggressiveness",
        "preserve_current_strategy",
        "observe_more",
    }
    assert score.reasons


def test_controller_strategy_advice_is_policy_gated_advisory_only() -> None:
    x = np.linspace(-1.0, 1.0, 32)[:, None]
    y = x * x
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=2, num_basis=8, seed=22),
        config=TrainerConfig(run_id="advice", epochs=3, learning_rate=0.05),
    )
    state = trainer.train(x, y)[-1]
    trace = trace_from_training_state("advice", state)
    score = TraceRanker().score(trace)

    profile = ControllerProfile("mc-stack")
    profile.update(score)
    advice = StrategyAdvisor().advise(profile)
    decision = ControllerUpdatePolicy(minimum_confidence=0.0).decide(advice)

    assert advice.controller_id == "mc-stack"
    assert decision.reason in {
        "accepted_as_advisory_only_no_mutation",
        "insufficient_trace_evidence",
        "confidence_below_policy_threshold",
    }
    assert "mutation" not in advice.to_dict()


def test_trainer_persists_trace_scores_and_strategy_advice() -> None:
    x = np.linspace(-1.0, 1.0, 48)[:, None]
    y = np.cos(np.pi * x)
    store = MemoryStore()
    trainer = PolicyGatedTrainer(
        BSplineLayer(1, 1, degree=3, num_basis=9, seed=23),
        store=store,
        config=TrainerConfig(run_id="v130", epochs=4, learning_rate=0.04),
    )
    trainer.train(x, y)

    base = "/experiments/v130"
    trace_score = store.read_json(f"{base}/evolution/trace_scores", "epoch_000004")
    advice = store.read_json(f"{base}/controllers/strategy_advice", "epoch_000004")
    decision = store.read_json(f"{base}/policy/controller_updates", "epoch_000004")
    memory = store.read_json(f"{base}/evolution/trace_memory", "summary")

    assert "rank_score" in trace_score
    assert advice["controller_id"] == "global_meta_controller_stack"
    assert decision["reason"] in {
        "accepted_as_advisory_only_no_mutation",
        "insufficient_trace_evidence",
        "confidence_below_policy_threshold",
    }
    assert memory["count"] == 4
