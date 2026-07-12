from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from pobsnn.policy import PolicyDecision, PolicyEngine, PolicyProposal, ProposalSeverity
from pobsnn.routing import ControllerRouteResponse

from .base import ControllerContext, MetaController
from .observers import (
    CompressionController,
    KnotUtilizationController,
    LossController,
    SmoothnessController,
    StabilityController,
)


_SEVERITY_CONFIDENCE = {
    ProposalSeverity.INFO: 0.55,
    ProposalSeverity.WATCH: 0.70,
    ProposalSeverity.CAUTION: 0.85,
    ProposalSeverity.CRITICAL: 1.00,
}


@dataclass(slots=True)
class MetaControllerStack:
    """Static controller stack plus v1.5 routed response execution."""

    controllers: list[MetaController] = field(default_factory=lambda: [
        LossController(),
        KnotUtilizationController(),
        SmoothnessController(),
        StabilityController(),
        CompressionController(),
    ])
    policy_engine: PolicyEngine = field(default_factory=PolicyEngine)

    def controller_map(self, extras: Mapping[str, MetaController] | None = None) -> dict[str, MetaController]:
        result = {controller.controller_id: controller for controller in self.controllers}
        if extras:
            result.update(extras)
        return result

    def observe(self, context: ControllerContext) -> tuple[list[PolicyProposal], list[PolicyDecision]]:
        proposals: list[PolicyProposal] = []
        for controller in self.controllers:
            proposals.extend(controller.observe(context))
        decisions = self.policy_engine.review(proposals)
        return proposals, decisions

    def respond_to_route(
        self,
        context: ControllerContext,
        targets: tuple[str, ...],
        *,
        recursive_ids: Mapping[str, str] | None = None,
        extras: Mapping[str, MetaController] | None = None,
    ) -> list[ControllerRouteResponse]:
        controller_map = self.controller_map(extras)
        responses: list[ControllerRouteResponse] = []
        for controller_id in targets:
            controller = controller_map.get(controller_id)
            if controller is None:
                responses.append(
                    ControllerRouteResponse(
                        controller_id=controller_id,
                        accepted=False,
                        confidence=0.0,
                        note="controller_not_registered",
                        recursive_id=None if recursive_ids is None else recursive_ids.get(controller_id),
                    )
                )
                continue
            proposals = controller.observe(context)
            if not proposals:
                responses.append(
                    ControllerRouteResponse(
                        controller_id=controller_id,
                        accepted=False,
                        confidence=0.25,
                        note="no_proposal_for_current_context",
                        recursive_id=None if recursive_ids is None else recursive_ids.get(controller_id),
                    )
                )
                continue
            leading = max(proposals, key=lambda p: _SEVERITY_CONFIDENCE[p.severity])
            responses.append(
                ControllerRouteResponse(
                    controller_id=controller_id,
                    accepted=True,
                    confidence=float(_SEVERITY_CONFIDENCE[leading.severity]),
                    note=leading.title,
                    proposal=leading.to_dict(),
                    recursive_id=None if recursive_ids is None else recursive_ids.get(controller_id),
                )
            )
        return responses
