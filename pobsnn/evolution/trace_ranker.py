from __future__ import annotations

from typing import Iterable

from .trace_score import TraceScore
from .traces import EvolutionTrace


class TraceRanker:
    """Deterministic ranker for POBSNN evolution traces.

    The ranker is intentionally simple in v1.3.0. It creates an auditable,
    hand-typeable advisory signal from already-observed training facts.
    """

    def score(self, trace: EvolutionTrace) -> TraceScore:
        quality = _clamp01(0.50 + trace.score_delta * 25.0)
        stability = _clamp01(1.0 - abs(trace.score_delta) * 10.0)
        compression = _clamp01(trace.compression_pressure)
        novelty = _clamp01(0.15 * len(set(trace.tags)))
        if trace.spawn_requested:
            novelty = _clamp01(novelty + 0.20)

        # Reward useful movement without encouraging reckless compression.
        rank = (
            0.40 * quality
            + 0.25 * stability
            + 0.20 * compression
            + 0.15 * novelty
        )
        reasons: list[str] = []
        if trace.score_delta > 0:
            reasons.append("validation_score_improved")
        elif trace.score_delta < 0:
            reasons.append("validation_score_declined")
        else:
            reasons.append("validation_score_flat")

        if trace.compression_pressure > 0.66:
            reasons.append("compression_pressure_high")
        elif trace.compression_pressure > 0.33:
            reasons.append("compression_pressure_medium")
        else:
            reasons.append("compression_pressure_low")

        if trace.spawn_requested:
            reasons.append("spawn_was_requested")

        action = self._recommend(trace, quality, stability, compression)
        return TraceScore(
            trace_id=trace.trace_id,
            rank_score=float(rank),
            quality_score=float(quality),
            stability_score=float(stability),
            compression_score=float(compression),
            novelty_score=float(novelty),
            recommended_action=action,
            reasons=reasons,
        )

    def rank(self, traces: Iterable[EvolutionTrace]) -> list[TraceScore]:
        scores = [self.score(t) for t in traces]
        return sorted(scores, key=lambda item: item.rank_score, reverse=True)

    @staticmethod
    def _recommend(trace: EvolutionTrace, quality: float, stability: float, compression: float) -> str:
        if compression > 0.66 and trace.score_delta > 0:
            return "prefer_observer_spawn_over_prune"
        if compression > 0.66 and trace.score_delta < 0:
            return "simulate_merge_or_freeze_before_action"
        if stability < 0.35:
            return "reduce_controller_aggressiveness"
        if quality > 0.60 and compression < 0.33:
            return "preserve_current_strategy"
        return "observe_more"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
