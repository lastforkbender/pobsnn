from __future__ import annotations

import numpy as np


def mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    err = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(err * err))
