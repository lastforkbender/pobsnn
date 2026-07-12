from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum


class NodeProperty(str, Enum):
    STABLE = "stable"
    REDUNDANT = "redundant"
    VOLATILE = "volatile"
    EMERGENT = "emergent"
    UNDERUTILIZED = "underutilized"


@dataclass(frozen=True)
class NodePropertyState:
    node_index: int
    property: NodeProperty
    utilization: float
    coefficient_norm: float
    svd_contribution: float
    score_delta: float
    compression_pressure: float
    rationale: str

    def to_dict(self) -> dict[str, int | float | str]:
        data = asdict(self)
        data["property"] = self.property.value
        return data
