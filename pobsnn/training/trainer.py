from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pobsnn.controllers import MetaControllerStack, SpawnArbiter, SpawnedObserver, ControllerProfile, StrategyAdvisor
from pobsnn.controllers.base import ControllerContext
from pobsnn.core import BSplineLayer
from pobsnn.node_properties import classify_nodes
from pobsnn.policy import ControllerUpdatePolicy
from pobsnn.recursion import compression_state_from_svd, evaluate_recursion_gate
from pobsnn.storage import BaseStore, MemoryStore
from pobsnn.telemetry import structural_telemetry, svd_summary
from pobsnn.evolution import TraceMemory, TraceRanker, trace_from_training_state

from .loss_functions import mse_loss
from .optimizers import FullBatchMSEOptimizer
from .snapshots import layer_snapshot
from .training_state import TrainingState


@dataclass(frozen=True)
class TrainerConfig:
    run_id: str = "run_001"
    epochs: int = 100
    learning_rate: float = 0.05
    snapshot_interval: int = 25
    persist_every_epoch: bool = True
    allow_spawned_observers: bool = True
    enable_trace_ranking: bool = True


class PolicyGatedTrainer:
    """Phase 3 integrated loop: training + telemetry + SVD + TDS memory.

    Coefficients are updated by the optimizer. Structural changes are not
    performed here. Meta-controllers observe and the policy engine records
    decisions; spawned controllers are observation-only.
    """

    def __init__(
        self,
        layer: BSplineLayer,
        *,
        store: BaseStore | None = None,
        config: TrainerConfig | None = None,
        controller_stack: MetaControllerStack | None = None,
    ) -> None:
        self.layer = layer
        self.store = store or MemoryStore()
        self.config = config or TrainerConfig()
        self.optimizer = FullBatchMSEOptimizer(self.config.learning_rate)
        self.controller_stack = controller_stack or MetaControllerStack()
        self.spawn_arbiter = SpawnArbiter()
        self.spawned: list[SpawnedObserver] = []
        self.history: list[TrainingState] = []
        self.trace_ranker = TraceRanker()
        self.trace_memory = TraceMemory()
        self.controller_profile = ControllerProfile("global_meta_controller_stack")
        self.strategy_advisor = StrategyAdvisor()
        self.controller_update_policy = ControllerUpdatePolicy()

    def train(self, x: np.ndarray, target: np.ndarray) -> list[TrainingState]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(target, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        previous_score: float | None = None
        previous_coefficients = self.layer.coefficients.copy()

        self._write_manifest()
        for epoch in range(1, self.config.epochs + 1):
            gradient = self.optimizer.step(self.layer, x_arr, y_arr)
            pred = self.layer(x_arr)
            loss = mse_loss(pred, y_arr)
            score = -loss
            score_delta = 0.0 if previous_score is None else score - previous_score

            telemetry = structural_telemetry(self.layer, x_arr, y_arr, previous_coefficients)
            svd = svd_summary(self.layer)
            compression = compression_state_from_svd(svd)
            nodes = classify_nodes(
                self.layer,
                x_arr,
                score_delta=score_delta,
                compression_pressure=compression.pressure,
            )
            gate = evaluate_recursion_gate(compression, nodes, score_delta=score_delta)

            context = ControllerContext(
                structural=telemetry,
                svd=svd,
            )
            proposals, decisions = self.controller_stack.observe(context)
            for observer in list(self.spawned):
                proposals.extend(observer.observe(context))
                if observer.expired:
                    self.spawned.remove(observer)

            spawn_proposal = None
            if self.config.allow_spawned_observers:
                spawn_proposal = self.spawn_arbiter.propose_spawn(gate, compression, nodes)
                if spawn_proposal is not None:
                    self.spawned.append(SpawnedObserver(spawn_proposal))

            state = TrainingState(
                epoch=epoch,
                loss=loss,
                score=score,
                score_delta=score_delta,
                svd=svd.to_dict(),
                compression=compression.to_dict(),
                node_states=[n.to_dict() for n in nodes],
                recursion_gate=gate.to_dict(),
                controller_proposals=[p.to_dict() for p in proposals],
                policy_decisions=[d.to_dict() for d in decisions],
                spawn_proposal=spawn_proposal.to_dict() if spawn_proposal else None,
                gradient_step=gradient.to_dict(),
            )
            self.history.append(state)
            if self.config.persist_every_epoch:
                self._persist_state(state, telemetry.to_dict())
            if epoch % self.config.snapshot_interval == 0 or epoch == self.config.epochs:
                self._persist_snapshot(epoch)

            previous_score = score
            previous_coefficients = self.layer.coefficients.copy()

        self.store.flush()
        return self.history

    def _base(self) -> str:
        return f"/experiments/{self.config.run_id}"

    def _write_manifest(self) -> None:
        self.store.write_json(self._base(), "manifest", {
            "project": "POBSNN",
            "phase": "v1.3.0-trace-ranked-controller-intelligence",
            "storage_semantics": "TDS VFS semantic memory adapter; TDS library remains separate, unmodified, and outside the reasoning region.",
            "reasoning_boundary": "POBSNN owns traces, scoring, policy, and aggregation. TDS stores serialized records only.",
            "cpu_only": True,
            "gpu_path": False,
            "run_id": self.config.run_id,
        })

    def _persist_state(self, state: TrainingState, telemetry: dict) -> None:
        epoch_key = f"epoch_{state.epoch:06d}"
        base = self._base()
        self.store.write_json(f"{base}/training/history", epoch_key, state.to_dict())
        self.store.write_json(f"{base}/telemetry/structural", epoch_key, telemetry)
        self.store.write_json(f"{base}/telemetry/svd", epoch_key, state.svd)
        self.store.write_json(f"{base}/controllers/reports", epoch_key, {
            "proposals": state.controller_proposals,
            "decisions": state.policy_decisions,
            "spawn": state.spawn_proposal,
        })
        self.store.write_json(f"{base}/recursion/gates", epoch_key, state.recursion_gate)
        self.store.write_json(f"{base}/node_states", epoch_key, state.node_states)
        trace = trace_from_training_state(self.config.run_id, state)
        self.store.write_json(f"{base}/evolution/traces", epoch_key, trace.to_dict())
        if self.config.enable_trace_ranking:
            trace_score = self.trace_ranker.score(trace)
            self.trace_memory.add(trace_score)
            self.controller_profile.update(trace_score)
            advice = self.strategy_advisor.advise(self.controller_profile)
            update_decision = self.controller_update_policy.decide(advice)
            self.store.write_json(f"{base}/evolution/trace_scores", epoch_key, trace_score.to_dict())
            self.store.write_json(f"{base}/controllers/strategy_advice", epoch_key, advice.to_dict())
            self.store.write_json(f"{base}/policy/controller_updates", epoch_key, update_decision.to_dict())
            self.store.write_json(f"{base}/evolution/trace_memory", "summary", self.trace_memory.to_dict())

    def _persist_snapshot(self, epoch: int) -> None:
        self.store.write_json(
            f"{self._base()}/network/snapshots",
            f"epoch_{epoch:06d}",
            layer_snapshot(self.layer),
        )
