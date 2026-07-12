from __future__ import annotations

from typing import Iterable

from .trace_score import TraceScore
from .traces import EvolutionTrace


class TraceRanker:
    """Transparent POBSNN source-score calculator.

    TDS Spiral may deterministically order these caller-supplied scores.  This
    class owns their meaning and deliberately avoids subjective novelty terms.
    """

    def score(self, trace: EvolutionTrace) -> TraceScore:
        quality = _clamp01(0.50 + trace.score_delta * 25.0)
        stability = _clamp01(1.0 - abs(trace.score_delta) * 10.0)
        compression = _clamp01(trace.compression_pressure)
        relational = _clamp01(
            0.45 * min(1.0, trace.interval_mean_distance)
            + 0.35 * min(1.0, trace.cognitive_step_length)
            + 0.20 * min(1.0, trace.route_trace_count / 4.0)
        )
        source_confidence = _clamp01(trace.source_confidence)

        # Relational movement has value only when it remains stable and is
        # accompanied by useful task movement.  It is not rewarded by itself.
        rank = (
            0.40 * quality
            + 0.25 * stability
            + 0.12 * compression
            + 0.13 * relational * stability
            + 0.10 * source_confidence
        )
        reasons: list[str] = []
        if trace.score_delta > 0:
            reasons.append("training_score_improved")
        elif trace.score_delta < 0:
            reasons.append("training_score_declined")
        else:
            reasons.append("training_score_flat")

        if trace.interval_mean_distance > 0.0:
            reasons.append("controller_interval_field_present")
        if trace.cognitive_step_length > 0.0:
            reasons.append("orthographic_process_movement_present")
        if trace.route_trace_count > 0:
            reasons.append("routed_communication_episode_present")
        if trace.max_recursive_depth > 0:
            reasons.append("recursive_descendant_evidence_present")

        action = self._recommend(trace, quality, stability, compression, relational)
        return TraceScore(
            trace_id=trace.trace_id,
            rank_score=float(_clamp01(rank)),
            quality_score=float(quality),
            stability_score=float(stability),
            compression_score=float(compression),
            relational_movement_score=float(relational),
            source_confidence=float(source_confidence),
            recursive_depth=int(trace.max_recursive_depth),
            recommended_action=action,
            reasons=tuple(reasons),
        )

    def rank(self, traces: Iterable[EvolutionTrace]) -> list[TraceScore]:
        scores = [self.score(t) for t in traces]
        return sorted(scores, key=lambda item: (item.rank_score, item.trace_id), reverse=True)

    @staticmethod
    def _recommend(
        trace: EvolutionTrace,
        quality: float,
        stability: float,
        compression: float,
        relational: float,
    ) -> str:
        if compression > 0.66 and trace.score_delta > 0:
            return "prefer_observer_spawn_over_prune"
        if compression > 0.66 and trace.score_delta < 0:
            return "simulate_merge_or_freeze_before_action"
        if stability < 0.35:
            return "reduce_controller_aggressiveness"
        if relational > 0.75 and quality < 0.5:
            return "preserve_interval_evidence_without_escalation"
        if quality > 0.60 and compression < 0.33:
            return "preserve_current_strategy"
        return "observe_more"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
