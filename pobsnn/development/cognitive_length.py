from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .controller_position import ControllerPosition, POSITION_AXES


@dataclass(frozen=True, slots=True)
class CognitiveLengthStep:
    """Dimensional process-length term tied to recursive identity.

    The name follows the project vocabulary.  The scalar measures movement in
    the declared orthographic coordinates; it is not intelligence, novelty, or
    correctness.
    """

    controller_id: str
    recursive_id: str
    epoch: int
    component_deltas: tuple[float, ...]
    step_length: float
    cumulative_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "recursive_id": self.recursive_id,
            "epoch": int(self.epoch),
            "axes": list(POSITION_AXES),
            "component_deltas": [float(v) for v in self.component_deltas],
            "step_length": float(self.step_length),
            "cumulative_length": float(self.cumulative_length),
        }


class CognitiveLengthTracker:
    """Keeps only the previous position and cumulative scalar per controller."""

    __slots__ = ("_previous", "_cumulative")

    def __init__(self) -> None:
        self._previous: dict[str, tuple[float, ...]] = {}
        self._cumulative: dict[str, float] = {}

    def update(self, positions: tuple[ControllerPosition, ...]) -> tuple[CognitiveLengthStep, ...]:
        steps: list[CognitiveLengthStep] = []
        for position in sorted(positions, key=lambda p: p.controller_id):
            current = position.vector(POSITION_AXES)
            previous = self._previous.get(position.controller_id, current)
            delta = tuple(float(c - p) for c, p in zip(current, previous))
            step_length = math.sqrt(sum(v * v for v in delta))
            cumulative = self._cumulative.get(position.controller_id, 0.0) + step_length
            self._previous[position.controller_id] = current
            self._cumulative[position.controller_id] = cumulative
            steps.append(
                CognitiveLengthStep(
                    controller_id=position.controller_id,
                    recursive_id=position.recursive_id,
                    epoch=position.epoch,
                    component_deltas=delta,
                    step_length=float(step_length),
                    cumulative_length=float(cumulative),
                )
            )
        return tuple(steps)
