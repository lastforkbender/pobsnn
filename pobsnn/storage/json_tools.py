from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert POBSNN values to TDS JSON-compatible data.

    TDS owns persistence; POBSNN normalizes NumPy/dataclass objects before
    handing them to the published TDS write_json API.
    """
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
