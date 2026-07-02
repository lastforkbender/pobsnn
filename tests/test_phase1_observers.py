import numpy as np

from pobsnn.controllers import ControllerContext, MetaControllerStack
from pobsnn.core import BSplineLayer
from pobsnn.telemetry import structural_telemetry, svd_summary


def test_structural_telemetry_is_finite():
    x = np.linspace(-1.0, 1.0, 32)[:, None]
    y = np.sin(np.pi * x)
    layer = BSplineLayer(1, 1, num_basis=8, seed=3)
    layer.fit_coefficients_mse(x, y, epochs=10, learning_rate=0.05)
    t = structural_telemetry(layer, x, target=y)
    assert t.numerical_finite
    assert t.active_basis_ratio > 0.0


def test_svd_summary_shape_and_rank():
    layer = BSplineLayer(2, 3, num_basis=5, seed=4)
    s = svd_summary(layer)
    assert s.full_rank == 3
    assert 1 <= s.effective_rank <= s.full_rank


def test_controller_stack_records_observations():
    x = np.linspace(-1.0, 1.0, 32)[:, None]
    y = np.sin(np.pi * x)
    layer = BSplineLayer(1, 1, num_basis=8, seed=5)
    t = structural_telemetry(layer, x, target=y)
    s = svd_summary(layer)
    proposals, decisions = MetaControllerStack().observe(ControllerContext(t, s))
    assert len(proposals) >= 4
    assert len(decisions) == len(proposals)
    assert all(not p.may_mutate_engine for p in proposals)
