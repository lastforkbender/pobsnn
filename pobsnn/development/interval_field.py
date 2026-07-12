from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

from .controller_position import ControllerPosition, POSITION_AXES


@dataclass(frozen=True, slots=True)
class SignedControllerInterval:
    controller_a: str
    controller_b: str
    recursive_id_a: str
    recursive_id_b: str
    axis_deltas: tuple[float, ...]
    l2_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_a": self.controller_a,
            "controller_b": self.controller_b,
            "recursive_id_a": self.recursive_id_a,
            "recursive_id_b": self.recursive_id_b,
            "axes": list(POSITION_AXES),
            "axis_deltas": [float(v) for v in self.axis_deltas],
            "l2_distance": float(self.l2_distance),
        }


@dataclass(frozen=True, slots=True)
class IntervalFieldSnapshot:
    """Relational population substrate for one process stride.

    The snapshot intentionally stops before entropy, abstraction, novelty, or
    cognitive-arrival semantics.  v1.5 establishes the auditable interval field
    those later mechanisms require.
    """

    epoch: int
    axes: tuple[str, ...]
    positions: tuple[ControllerPosition, ...]
    intervals: tuple[SignedControllerInterval, ...]
    mean_pair_distance: float
    max_pair_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "axes": list(self.axes),
            "positions": [p.to_dict() for p in self.positions],
            "intervals": [i.to_dict() for i in self.intervals],
            "pair_count": len(self.intervals),
            "mean_pair_distance": float(self.mean_pair_distance),
            "max_pair_distance": float(self.max_pair_distance),
        }


def build_interval_field(
    positions: tuple[ControllerPosition, ...],
    *,
    epoch: int,
) -> IntervalFieldSnapshot:
    ordered = tuple(sorted(positions, key=lambda p: p.controller_id))
    intervals: list[SignedControllerInterval] = []
    for left, right in combinations(ordered, 2):
        lv = left.vector(POSITION_AXES)
        rv = right.vector(POSITION_AXES)
        delta = tuple(float(b - a) for a, b in zip(lv, rv))
        distance = math.sqrt(sum(v * v for v in delta))
        intervals.append(
            SignedControllerInterval(
                controller_a=left.controller_id,
                controller_b=right.controller_id,
                recursive_id_a=left.recursive_id,
                recursive_id_b=right.recursive_id,
                axis_deltas=delta,
                l2_distance=float(distance),
            )
        )
    distances = [item.l2_distance for item in intervals]
    mean_distance = 0.0 if not distances else sum(distances) / len(distances)
    max_distance = 0.0 if not distances else max(distances)
    return IntervalFieldSnapshot(
        epoch=int(epoch),
        axes=POSITION_AXES,
        positions=ordered,
        intervals=tuple(intervals),
        mean_pair_distance=float(mean_distance),
        max_pair_distance=float(max_distance),
    )
