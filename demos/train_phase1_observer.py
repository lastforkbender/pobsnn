from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pobsnn.controllers import ControllerContext, MetaControllerStack
from pobsnn.core import BSplineLayer
from pobsnn.telemetry import structural_telemetry, svd_summary


def main() -> None:
    x = np.linspace(-1.0, 1.0, 256)[:, None]
    y = (np.sin(np.pi * x[:, 0]) + 0.25 * np.cos(3.0 * np.pi * x[:, 0]))[:, None]

    layer = BSplineLayer(1, 1, degree=3, num_basis=16, seed=7)
    previous = layer.coefficients.copy()
    losses = layer.fit_coefficients_mse(x, y, epochs=800, learning_rate=0.08)

    structural = structural_telemetry(layer, x, target=y, previous_coefficients=previous)
    svd = svd_summary(layer)
    stack = MetaControllerStack()
    proposals, decisions = stack.observe(ControllerContext(structural=structural, svd=svd))

    report = {
        "phase": "1-observation",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "structural": structural.to_dict(),
        "svd": svd.to_dict(),
        "proposals": [p.to_dict() for p in proposals],
        "decisions": [d.to_dict() for d in decisions],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
