from __future__ import annotations

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.node_properties import classify_nodes
from pobsnn.recursion import compression_state_from_svd, evaluate_recursion_gate
from pobsnn.telemetry import svd_summary, structural_telemetry
from pobsnn.controllers.spawning import SpawnArbiter, SpawnedObserver


def main() -> None:
    x = np.linspace(-1.0, 1.0, 160)[:, None]
    y = (np.sin(np.pi * x[:, 0]) + 0.15 * np.cos(3 * np.pi * x[:, 0]))[:, None]

    layer = BSplineLayer(in_features=1, out_features=8, num_basis=14, seed=7)
    # make several output nodes deliberately redundant for demonstration
    layer.coefficients[3:] = layer.coefficients[0] * 0.02
    losses = layer.fit_coefficients_mse(x, np.repeat(y, 8, axis=1), epochs=80, learning_rate=0.03)

    svd = svd_summary(layer)
    compression = compression_state_from_svd(svd)
    node_states = classify_nodes(layer, x, score_delta=losses[-1] - losses[0], compression_pressure=compression.pressure)
    gate = evaluate_recursion_gate(compression, node_states, score_delta=losses[-1] - losses[0])

    arbiter = SpawnArbiter()
    proposal = arbiter.propose_spawn(gate, compression, node_states)

    print("Phase 2 spawning demo")
    print("loss_start", round(losses[0], 6), "loss_end", round(losses[-1], 6))
    print("svd", svd.to_dict())
    print("compression", compression.to_dict())
    print("gate", gate.to_dict())
    print("typed_nodes", [s.to_dict() for s in node_states])
    if proposal is not None:
        child = SpawnedObserver(proposal)
        print("spawn_proposal", proposal.to_dict())
        print("spawned_observer", child.observe(None)[0].to_dict())
    else:
        print("spawn_proposal", None)


if __name__ == "__main__":
    main()
