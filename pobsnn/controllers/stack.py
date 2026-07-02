from __future__ import annotations

from dataclasses import dataclass, field

from pobsnn.policy import PolicyDecision, PolicyEngine, PolicyProposal

from .base import ControllerContext, MetaController
from .observers import (
    CompressionController,
    KnotUtilizationController,
    LossController,
    SmoothnessController,
    StabilityController,
)


@dataclass
class MetaControllerStack:
    """Static Phase 1 controller stack.

    No spawning yet. Controllers observe, emit proposals, and pass them to a
    policy engine that blocks mutation by default.
    """

    controllers: list[MetaController] = field(default_factory=lambda: [
        LossController(),
        KnotUtilizationController(),
        SmoothnessController(),
        StabilityController(),
        CompressionController(),
    ])
    policy_engine: PolicyEngine = field(default_factory=PolicyEngine)

    def observe(self, context: ControllerContext) -> tuple[list[PolicyProposal], list[PolicyDecision]]:
        proposals: list[PolicyProposal] = []
        for controller in self.controllers:
            proposals.extend(controller.observe(context))
        decisions = self.policy_engine.review(proposals)
        return proposals, decisions
