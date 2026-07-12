from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from pobsnn.controllers import (
    ControllerProfile,
    MetaControllerStack,
    SpawnArbiter,
    SpawnedObserver,
    StrategyAdvisor,
)
from pobsnn.controllers.base import ControllerContext
from pobsnn.core import BSplineLayer
from pobsnn.development import (
    CognitiveLengthTracker,
    CommunicationEpisode,
    RecursiveIdentity,
    build_interval_field,
    positions_from_proposals,
)
from pobsnn.evolution import TraceMemory, TraceRanker, trace_from_training_state
from pobsnn.identity import stable_digest
from pobsnn.node_properties import classify_nodes
from pobsnn.policy import ControllerUpdatePolicy, PolicyProposal, ProposalSeverity
from pobsnn.recursion import compression_state_from_svd, evaluate_recursion_gate
from pobsnn.routing import AggregationBus, ControllerRouter, RouteTrace, make_message
from pobsnn.storage import BaseStore, MemoryStore
from pobsnn.storage.residual_csv import episode_csv_families
from pobsnn.telemetry import structural_telemetry, svd_summary

from .loss_functions import mse_loss
from .optimizers import FullBatchMSEOptimizer, GradientStep
from .snapshots import layer_snapshot
from .training_state import TrainingState


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SEVERITY_VALUE = {
    ProposalSeverity.INFO: 0.15,
    ProposalSeverity.WATCH: 0.45,
    ProposalSeverity.CAUTION: 0.70,
    ProposalSeverity.CRITICAL: 1.00,
}


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    run_id: str = "run_001"
    epochs: int = 100
    learning_rate: float = 0.05
    snapshot_interval: int = 25
    persist_every_epoch: bool = True
    persist_csv_residuals: bool = True
    allow_spawned_observers: bool = True
    enable_trace_ranking: bool = True
    enable_tds_trace_ordering: bool = True
    halt_on_rejected_update: bool = True
    history_limit: int = 1024
    trace_memory_limit: int = 1024
    router_history_limit: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("run_id must be a non-empty artifact-safe identifier")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if self.history_limit <= 0:
            raise ValueError("history_limit must be positive")
        if self.trace_memory_limit <= 0:
            raise ValueError("trace_memory_limit must be positive")
        if self.router_history_limit <= 0:
            raise ValueError("router_history_limit must be positive")


class PolicyGatedTrainer:
    """POBSNN v1.5 recursive-evidence training loop.

    v1.5 remains a fixed-knot additive B-spline research substrate.  It adds
    mathematical hardening, trainer-integrated communication episodes,
    deterministic recursive source identities, controller interval evidence,
    bounded process-length terms, and TDS v3.5.2 artifact integration.

    Entropy semantics, origamic unions, prospective cognitive arrival, and
    mature personality are explicitly reserved for later releases.
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
        self.history: deque[TrainingState] = deque(maxlen=self.config.history_limit)
        self.trace_ranker = TraceRanker()
        self.trace_memory = TraceMemory(self.config.trace_memory_limit)
        self.controller_profile = ControllerProfile("global_meta_controller_stack")
        self.strategy_advisor = StrategyAdvisor()
        self.controller_update_policy = ControllerUpdatePolicy()
        self.router = ControllerRouter(max_history=self.config.router_history_limit)
        self.aggregation_bus = AggregationBus()
        self.cognitive_length_tracker = CognitiveLengthTracker()
        self.last_tds_rank_run: dict[str, Any] | None = None

    def train(self, x: np.ndarray, target: np.ndarray) -> list[TrainingState]:
        x_arr, y_arr = self._validate_training_arrays(x, target)
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
            context = ControllerContext(structural=telemetry, svd=svd)

            proposals, decisions = self.controller_stack.observe(context)
            arbiter_proposals = self.spawn_arbiter.observe(context)
            proposals.extend(arbiter_proposals)
            decisions.extend(self.controller_stack.policy_engine.review(arbiter_proposals))

            for observer in list(self.spawned):
                observer_proposals = observer.observe(context)
                proposals.extend(observer_proposals)
                decisions.extend(self.controller_stack.policy_engine.review(observer_proposals))
                if observer.expired:
                    self.spawned.remove(observer)

            if not gradient.accepted:
                rejected = PolicyProposal(
                    controller_id="MC-3",
                    severity=ProposalSeverity.CRITICAL,
                    title="Numerical update rejected",
                    rationale=gradient.rejection_reason,
                    recommended_action="Training is halted after transactional rollback.",
                    may_mutate_engine=False,
                )
                proposals.append(rejected)
                decisions.extend(self.controller_stack.policy_engine.review([rejected]))

            spawn_proposal = None
            if self.config.allow_spawned_observers:
                spawn_proposal = self.spawn_arbiter.propose_spawn(gate, compression, nodes)
                if spawn_proposal is not None:
                    self.spawned.append(SpawnedObserver(spawn_proposal))

            episode_id = "EP-" + stable_digest("communication-episode", self.config.run_id, epoch)
            stride_id = "ST-" + stable_digest("process-stride", self.config.run_id, epoch, score_delta)
            identities, identity_map = self._build_recursive_identities(
                episode_id,
                proposals,
                spawn_proposal=spawn_proposal,
            )
            positions = positions_from_proposals(proposals, identity_map, epoch=epoch)
            interval_field = build_interval_field(positions, epoch=epoch)
            length_steps = self.cognitive_length_tracker.update(positions)

            route_traces = self._execute_routes(
                epoch=epoch,
                context=context,
                telemetry=telemetry,
                compression=compression,
                gate=gate,
                proposals=proposals,
                identity_map=identity_map,
                spawn_proposal=spawn_proposal,
                gradient=gradient,
            )

            episode = CommunicationEpisode(
                episode_id=episode_id,
                stride_id=stride_id,
                run_id=self.config.run_id,
                epoch=epoch,
                recursive_identities=identities,
                route_traces=tuple(route_traces),
                interval_field=interval_field,
                cognitive_length=length_steps,
            )

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
                route_traces=[trace.to_dict() for trace in route_traces],
                communication_episode=episode.to_dict(),
                halted=not gradient.accepted,
                halt_reason=gradient.rejection_reason,
            )

            state = self._reason_about_state(state, context, identity_map)
            self.history.append(state)

            if self.config.persist_every_epoch:
                self._persist_state(state, telemetry.to_dict())
            if epoch % self.config.snapshot_interval == 0 or epoch == self.config.epochs or state.halted:
                self._persist_snapshot(epoch)

            previous_score = score
            previous_coefficients = self.layer.coefficients.copy()
            if state.halted and self.config.halt_on_rejected_update:
                break

        self.store.flush()
        return list(self.history)

    @staticmethod
    def _validate_training_arrays(x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(target, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        if x_arr.ndim != 2 or y_arr.ndim != 2:
            raise ValueError("x and target must be one- or two-dimensional arrays")
        if x_arr.shape[0] == 0 or y_arr.shape[0] == 0:
            raise ValueError("x and target must contain at least one sample")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x and target must contain the same number of samples")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
            raise ValueError("x and target must contain only finite values")
        return x_arr, y_arr

    def _build_recursive_identities(
        self,
        episode_id: str,
        proposals: Iterable[PolicyProposal],
        *,
        spawn_proposal: Any | None,
    ) -> tuple[tuple[RecursiveIdentity, ...], dict[str, RecursiveIdentity]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for proposal in proposals:
            grouped.setdefault(proposal.controller_id, []).append(proposal.to_dict())

        identities: dict[str, RecursiveIdentity] = {}
        for controller_id in sorted(grouped):
            if controller_id.startswith("MC-5."):
                continue
            identities[controller_id] = RecursiveIdentity.root(
                self.config.run_id,
                episode_id,
                controller_id,
                source_params={"proposals": grouped[controller_id]},
            )

        if "MC-5" not in identities:
            identities["MC-5"] = RecursiveIdentity.root(
                self.config.run_id,
                episode_id,
                "MC-5",
                source_params={"proposals": grouped.get("MC-5", [])},
            )

        branch_index = 0
        for controller_id in sorted(grouped):
            if not controller_id.startswith("MC-5."):
                continue
            identities[controller_id] = identities["MC-5"].child(
                controller_id,
                branch_index=branch_index,
                source_params={"proposals": grouped[controller_id]},
            )
            branch_index += 1

        if spawn_proposal is not None and spawn_proposal.child_controller_id not in identities:
            identities[spawn_proposal.child_controller_id] = identities["MC-5"].child(
                spawn_proposal.child_controller_id,
                branch_index=branch_index,
                source_params=spawn_proposal.to_dict(),
            )

        ordered = tuple(identities[key] for key in sorted(identities))
        return ordered, identities

    def _execute_routes(
        self,
        *,
        epoch: int,
        context: ControllerContext,
        telemetry: Any,
        compression: Any,
        gate: Any,
        proposals: list[PolicyProposal],
        identity_map: dict[str, RecursiveIdentity],
        spawn_proposal: Any | None,
        gradient: GradientStep,
    ) -> list[RouteTrace]:
        max_severity = max((_SEVERITY_VALUE[p.severity] for p in proposals), default=0.0)
        specs: list[tuple[str, str, float, dict[str, Any], tuple[str, ...]]] = [
            (
                "PolicyReviewRequest",
                "trainer",
                max_severity,
                {"proposal_count": len(proposals), "recursion_decision": gate.decision.value},
                ("stride", "policy"),
            )
        ]
        if compression.pressure > 0.0:
            specs.append((
                "CompressionAlert",
                "svd_metrics",
                float(compression.pressure),
                compression.to_dict(),
                ("compression",),
            ))
        curvature_severity = min(1.0, float(telemetry.curvature_max_abs) / 0.25)
        if curvature_severity >= 0.25:
            specs.append((
                "CurvatureDriftAlert",
                "structural_metrics",
                curvature_severity,
                {
                    "curvature_max_abs": telemetry.curvature_max_abs,
                    "curvature_mean_abs": telemetry.curvature_mean_abs,
                },
                ("curvature",),
            ))
        if not telemetry.numerical_finite or telemetry.coefficient_l2 > 100.0:
            specs.append((
                "NodeInstabilityAlert",
                "structural_metrics",
                1.0 if not telemetry.numerical_finite else min(1.0, telemetry.coefficient_l2 / 200.0),
                {
                    "numerical_finite": telemetry.numerical_finite,
                    "coefficient_l2": telemetry.coefficient_l2,
                },
                ("stability",),
            ))
        if not gradient.accepted:
            specs.append((
                "NumericalUpdateRejected",
                "optimizer",
                1.0,
                gradient.to_dict(),
                ("critical", "rollback"),
            ))
        if spawn_proposal is not None:
            specs.append((
                "SpawnCandidate",
                "spawn_arbiter",
                min(1.0, float(compression.pressure)),
                spawn_proposal.to_dict(),
                ("spawn", "observation-only"),
            ))

        recursive_ids = {key: value.recursive_id for key, value in identity_map.items()}
        extras = {"MC-5": self.spawn_arbiter}
        traces: list[RouteTrace] = []
        for event_index, (message_type, source, severity, payload, tags) in enumerate(specs):
            message = make_message(
                message_type,
                source,
                severity=severity,
                epoch=epoch,
                event_index=event_index,
                payload=payload,
                tags=tags,
            )
            targets = self.router.targets_for(message)
            responses = self.controller_stack.respond_to_route(
                context,
                targets,
                recursive_ids=recursive_ids,
                extras=extras,
            )
            trace = self.aggregation_bus.aggregate(message, targets, responses)
            self.router.record(trace)
            traces.append(trace)
        return traces

    def _reason_about_state(
        self,
        state: TrainingState,
        context: ControllerContext,
        identity_map: dict[str, RecursiveIdentity],
    ) -> TrainingState:
        trace = trace_from_training_state(self.config.run_id, state)
        trace_score = None
        advice = None
        update_decision = None
        rank_run = None

        if self.config.enable_trace_ranking:
            trace_score = self.trace_ranker.score(trace)
            self.trace_memory.add(trace_score)
            self.controller_profile.update(trace_score)
            advice = self.strategy_advisor.advise(self.controller_profile)
            update_decision = self.controller_update_policy.decide(advice)

            if self.config.enable_tds_trace_ordering and self.store.supports_trace_ordering:
                scores = list(self.trace_memory.scores)
                rank_run = self.store.rank_trace_evidence(
                    [item.trace_id for item in scores],
                    [item.rank_score for item in scores],
                    confidences=[item.source_confidence for item in scores],
                    depths=[item.recursive_depth for item in scores],
                    ages_ns=[0 for _ in scores],
                    limit=min(64, len(scores)),
                )
                self.last_tds_rank_run = rank_run

            # Trace ranking itself becomes a routed result, but it is not fed
            # back into this same source score; that avoids a circular stride.
            message = make_message(
                "TraceRankingUpdate",
                "trace_ranker",
                severity=trace_score.rank_score,
                epoch=state.epoch,
                event_index=999,
                payload={
                    "trace_id": trace.trace_id,
                    "source_rank_score": trace_score.rank_score,
                    "recommended_action": trace_score.recommended_action,
                },
                tags=("trace-ranking", "post-stride"),
            )
            targets = self.router.targets_for(message)
            recursive_ids = {key: value.recursive_id for key, value in identity_map.items()}
            responses = self.controller_stack.respond_to_route(
                context,
                targets,
                recursive_ids=recursive_ids,
                extras={"MC-5": self.spawn_arbiter},
            )
            ranking_route = self.aggregation_bus.aggregate(message, targets, responses)
            self.router.record(ranking_route)
            route_traces = list(state.route_traces) + [ranking_route.to_dict()]
            episode = dict(state.communication_episode or {})
            episode["route_traces"] = route_traces
            state = replace(state, route_traces=route_traces, communication_episode=episode)

        return replace(
            state,
            evolution_trace=trace.to_dict(),
            trace_score=None if trace_score is None else trace_score.to_dict(),
            tds_rank_run=rank_run,
            strategy_advice=None if advice is None else advice.to_dict(),
            controller_update=None if update_decision is None else update_decision.to_dict(),
        )

    def _base(self) -> str:
        return f"/experiments/{self.config.run_id}"

    def _write_manifest(self) -> None:
        self.store.write_json(self._base(), "manifest", {
            "project_code_name": "POBSNN",
            "project_name_status": "working_code_name",
            "release": "1.5.0",
            "phase": "v1.5.0-recursive-evidence-substrate",
            "storage_semantics": "TDS v3.5.2 may preserve, validate, replay, and deterministically order supplied evidence; it remains outside reasoning.",
            "reasoning_boundary": "POBSNN owns source semantics, recursive identities, controller positions, intervals, process length, and policy decisions.",
            "artifact_integrity": "bounded live state; immutable stride evidence; later outcomes create new ranking runs rather than rewriting source artifacts",
            "maturity_boundary": {
                "implemented": [
                    "mathematical_hardening",
                    "trainer_integrated_routing",
                    "recursive_source_identity",
                    "controller_interval_substrate",
                    "bounded_cognitive_length_terms",
                    "tds_v3_5_2_result_first_csv_spiral_adapter",
                ],
                "reserved": [
                    "discrete_premodel_fold_orientations",
                    "origamic_unions",
                    "sine_cosine_manifestation",
                    "interval_entropy_abstraction",
                    "prospective_cognitive_arrival",
                    "mature_personality_perspective",
                ],
            },
            "cpu_only": True,
            "gpu_path": False,
            "run_id": self.config.run_id,
        }, overwrite=True)

    def _persist_state(self, state: TrainingState, telemetry: dict[str, Any]) -> None:
        epoch_key = f"epoch_{state.epoch:06d}"
        base = self._base()
        self.store.write_json(f"{base}/training/history", epoch_key, state.to_dict(), overwrite=False)
        self.store.write_json(f"{base}/telemetry/structural", epoch_key, telemetry, overwrite=False)
        self.store.write_json(f"{base}/telemetry/svd", epoch_key, state.svd, overwrite=False)
        self.store.write_json(f"{base}/controllers/reports", epoch_key, {
            "proposals": state.controller_proposals,
            "decisions": state.policy_decisions,
            "spawn": state.spawn_proposal,
        }, overwrite=False)
        self.store.write_json(f"{base}/recursion/gates", epoch_key, state.recursion_gate, overwrite=False)
        self.store.write_json(f"{base}/node_states", epoch_key, state.node_states, overwrite=False)
        self.store.write_json(f"{base}/communication/episodes", epoch_key, state.communication_episode, overwrite=False)
        for route in state.route_traces:
            self.store.write_json(
                f"{base}/communication/routes",
                str(route["trace_id"]),
                route,
                overwrite=False,
            )
        if state.evolution_trace is not None:
            self.store.write_json(f"{base}/evolution/traces", epoch_key, state.evolution_trace, overwrite=False)
        if state.trace_score is not None:
            self.store.write_json(f"{base}/evolution/trace_scores", epoch_key, state.trace_score, overwrite=False)
            self.store.write_json(f"{base}/evolution/trace_memory", "summary", self.trace_memory.to_dict(), overwrite=True)
        if state.strategy_advice is not None:
            self.store.write_json(f"{base}/controllers/strategy_advice", epoch_key, state.strategy_advice, overwrite=False)
        if state.controller_update is not None:
            self.store.write_json(f"{base}/policy/controller_updates", epoch_key, state.controller_update, overwrite=False)
        if state.tds_rank_run is not None:
            self.store.write_json(f"{base}/evolution/tds_rank_runs", epoch_key, state.tds_rank_run, overwrite=False)

        if self.config.persist_csv_residuals and self.store.supports_csv_artifacts and state.communication_episode:
            csv_base = f"{base}/evidence/stride_{state.epoch:06d}"
            for family, (fieldnames, rows) in episode_csv_families(state.communication_episode).items():
                csv_id = f"e{state.epoch:06d}_{family}"
                report = self.store.write_csv_artifact(
                    csv_base,
                    csv_id,
                    rows,
                    fieldnames=fieldnames,
                    overwrite=False,
                )
                self.store.write_json(
                    f"{csv_base}/reports",
                    f"{csv_id}_commit",
                    report,
                    overwrite=False,
                )

    def _persist_snapshot(self, epoch: int) -> None:
        self.store.write_json(
            f"{self._base()}/network/snapshots",
            f"epoch_{epoch:06d}",
            layer_snapshot(self.layer),
            overwrite=True,
        )
