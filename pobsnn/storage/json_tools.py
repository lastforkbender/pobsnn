from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert POBSNN values to strict JSON-compatible data."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")
