from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from pobsnn.policy import PolicyProposal, ProposalSeverity

from .recursive_identity import RecursiveIdentity


POSITION_AXES: tuple[str, ...] = (
    "loss",
    "basis_utilization",
    "curvature",
    "stability",
    "compression",
    "recursion",
)

CONTROLLER_AXIS: Mapping[str, str] = {
    "MC-0": "loss",
    "MC-1": "basis_utilization",
    "MC-2": "curvature",
    "MC-3": "stability",
    "MC-4": "compression",
    "MC-5": "recursion",
}

_SEVERITY_POSITION: Mapping[ProposalSeverity, float] = {
    ProposalSeverity.INFO: 0.15,
    ProposalSeverity.WATCH: 0.45,
    ProposalSeverity.CAUTION: 0.70,
    ProposalSeverity.CRITICAL: 1.00,
}


@dataclass(frozen=True, slots=True)
class AxisCoordinate:
    axis_id: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControllerPosition:
    """One meta-controller's transparent orthographic position.

    In v1.5 the position is a concern/evidence coordinate derived directly from
    emitted proposals.  It is not a novelty score and it is not yet a mature
    semantic vote.  Keeping the complete axis vector creates an integrity-safe
    substrate for later recursive treatment.
    """

    controller_id: str
    recursive_id: str
    epoch: int
    coordinates: tuple[AxisCoordinate, ...]
    confidence: float
    source_titles: tuple[str, ...]

    def vector(self, axes: tuple[str, ...] = POSITION_AXES) -> tuple[float, ...]:
        values = {item.axis_id: float(item.value) for item in self.coordinates}
        return tuple(values.get(axis, 0.0) for axis in axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "recursive_id": self.recursive_id,
            "epoch": self.epoch,
            "coordinates": [item.to_dict() for item in self.coordinates],
            "confidence": float(self.confidence),
            "source_titles": list(self.source_titles),
        }


def positions_from_proposals(
    proposals: Iterable[PolicyProposal],
    identities: Mapping[str, RecursiveIdentity],
    *,
    epoch: int,
) -> tuple[ControllerPosition, ...]:
    grouped: dict[str, list[PolicyProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.controller_id, []).append(proposal)

    positions: list[ControllerPosition] = []
    for controller_id in sorted(grouped):
        controller_proposals = grouped[controller_id]
        identity = identities.get(controller_id)
        if identity is None:
            continue
        highest = max(_SEVERITY_POSITION[p.severity] for p in controller_proposals)
        axis = CONTROLLER_AXIS.get(controller_id.split(".", 1)[0], "recursion")
        coordinates = tuple(
            AxisCoordinate(axis_id=item, value=(highest if item == axis else 0.0))
            for item in POSITION_AXES
        )
        # Confidence describes evidence clarity, not semantic truth.
        confidence = min(1.0, 0.50 + 0.50 * highest)
        positions.append(
            ControllerPosition(
                controller_id=controller_id,
                recursive_id=identity.recursive_id,
                epoch=int(epoch),
                coordinates=coordinates,
                confidence=float(confidence),
                source_titles=tuple(p.title for p in controller_proposals),
            )
        )
    return tuple(positions)
