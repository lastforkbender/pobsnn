from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pobsnn.core import BSplineLayer
from pobsnn.storage import MemoryStore, TDSVFSStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train POBSNN v1 and persist semantic memory.")
    parser.add_argument("--store", choices=["memory", "tds"], default="memory")
    parser.add_argument("--mount", default="./pobsnn_tds_mount")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    x = np.linspace(-1.0, 1.0, 160)[:, None]
    y = 0.65 * np.sin(np.pi * x) + 0.25 * np.cos(2.0 * np.pi * x)

    layer = BSplineLayer(1, 1, degree=3, num_basis=18, seed=11)
    store = MemoryStore() if args.store == "memory" else TDSVFSStore(Path(args.mount))

    trainer = PolicyGatedTrainer(
        layer,
        store=store,
        config=TrainerConfig(
            run_id="v1_curve_demo",
            epochs=args.epochs,
            learning_rate=0.07,
            snapshot_interval=max(1, args.epochs // 4),
        ),
    )
    history = trainer.train(x, y)
    print(f"epochs={len(history)}")
    print(f"first_loss={history[0].loss:.8f}")
    print(f"final_loss={history[-1].loss:.8f}")
    print(f"final_svd={history[-1].svd}")
    print(f"mount={args.mount if args.store == 'tds' else 'memory'}")


if __name__ == "__main__":
    main()
