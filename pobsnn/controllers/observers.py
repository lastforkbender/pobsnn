from __future__ import annotations

import math

from pobsnn.policy import PolicyProposal, ProposalSeverity

from .base import ControllerContext, MetaController


class LossController(MetaController):
    controller_id = "MC-0"
    name = "Primordial Loss Observer"

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        t = context.structural
        if math.isnan(t.mse_loss):
            return [PolicyProposal(self.controller_id, ProposalSeverity.INFO, "No target loss", "No target was supplied, so MSE is unavailable.", "Continue structural observation only.")]
        if t.mse_loss > 0.05:
            return [PolicyProposal(self.controller_id, ProposalSeverity.WATCH, "Loss remains high", f"MSE is {t.mse_loss:.6f}.", "Train longer or inspect basis resolution.")]
        return [PolicyProposal(self.controller_id, ProposalSeverity.INFO, "Loss stable", f"MSE is {t.mse_loss:.6f}.", "Continue observation.")]


class KnotUtilizationController(MetaController):
    controller_id = "MC-1"
    name = "Axiom Knot Utilization Observer"

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        ratio = context.structural.active_basis_ratio
        if ratio < 0.60:
            return [PolicyProposal(self.controller_id, ProposalSeverity.CAUTION, "Low basis utilization", f"Only {ratio:.2%} of basis functions are active.", "Consider wider sampling or future knot adaptation proposal.")]
        return [PolicyProposal(self.controller_id, ProposalSeverity.INFO, "Basis utilization healthy", f"Active basis ratio is {ratio:.2%}.", "Continue observation.")]


class SmoothnessController(MetaController):
    controller_id = "MC-2"
    name = "Harmonic Smoothness Observer"

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        curv = context.structural.curvature_max_abs
        if curv > 0.25:
            return [PolicyProposal(self.controller_id, ProposalSeverity.WATCH, "High local curvature", f"Curvature proxy max is {curv:.6f}.", "Inspect oscillation and consider regularization in a future phase.")]
        return [PolicyProposal(self.controller_id, ProposalSeverity.INFO, "Curvature stable", f"Curvature proxy max is {curv:.6f}.", "Continue observation.")]


class StabilityController(MetaController):
    controller_id = "MC-3"
    name = "Topologic Stability Observer"

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        t = context.structural
        if not t.numerical_finite:
            return [PolicyProposal(self.controller_id, ProposalSeverity.CRITICAL, "Numerical instability", "Outputs or coefficients contain non-finite values.", "Stop training and inspect learning rate.")]
        if t.coefficient_l2 > 100.0:
            return [PolicyProposal(self.controller_id, ProposalSeverity.CAUTION, "Coefficient magnitude high", f"Coefficient L2 is {t.coefficient_l2:.6f}.", "Lower learning rate or add coefficient regularization later.")]
        return [PolicyProposal(self.controller_id, ProposalSeverity.INFO, "Numerics stable", "All observed values are finite and coefficient magnitude is bounded.", "Continue observation.")]


class CompressionController(MetaController):
    controller_id = "MC-4"
    name = "Recursive SVD Observer"

    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        s = context.svd
        if s is None:
            return []
        if s.effective_rank < s.full_rank and s.energy_95_rank < s.full_rank:
            return [PolicyProposal(self.controller_id, ProposalSeverity.WATCH, "Possible low-rank redundancy", f"95% energy rank is {s.energy_95_rank} of {s.full_rank}.", "Record compression candidate only; do not compress in Phase 1.")]
        return [PolicyProposal(self.controller_id, ProposalSeverity.INFO, "No compression pressure", f"Effective rank is {s.effective_rank} of {s.full_rank}.", "Continue SVD observation.")]
