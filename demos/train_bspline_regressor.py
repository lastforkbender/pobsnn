from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pobsnn.core import BSplineLayer
from pobsnn.telemetry import spline_health


def main() -> None:
    x = np.linspace(-1.0, 1.0, 256)[:, None]
    y = (np.sin(np.pi * x[:, 0]) + 0.25 * np.cos(3.0 * np.pi * x[:, 0]))[:, None]

    layer = BSplineLayer(
        in_features=1,
        out_features=1,
        degree=3,
        num_basis=16,
        domain_min=-1.0,
        domain_max=1.0,
        seed=7,
    )

    losses = layer.fit_coefficients_mse(x, y, epochs=800, learning_rate=0.08)
    print(f"initial_loss={losses[0]:.8f}")
    print(f"final_loss={losses[-1]:.8f}")
    print(spline_health(layer, x).to_dict())

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pred = layer(x)
    plt.figure()
    plt.plot(x[:, 0], y[:, 0], label="target")
    plt.plot(x[:, 0], pred[:, 0], label="pobsnn")
    plt.legend()
    plt.title("POBSNN Phase 0 CPU B-spline Regressor")
    plt.show()


if __name__ == "__main__":
    main()
