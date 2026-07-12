from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Return a deterministic UTF-8 JSON representation.

    The helper is intentionally strict and small.  It is used only for stable
    identifiers and artifact fingerprints; application data is still stored
    through the configured storage backend.
    """

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_digest(namespace: str, *parts: Any, length: int = 24) -> str:
    """Create a process-stable identifier from canonical fields.

    ``length`` is the number of hexadecimal characters retained.  Twenty-four
    characters gives a 96-bit identifier, which is ample for the bounded local
    identities used by this research repository while keeping filenames short.
    """

    if not namespace or not namespace.strip():
        raise ValueError("namespace must not be empty")
    if length < 16 or length > 64:
        raise ValueError("length must be between 16 and 64 hexadecimal characters")
    payload = canonical_json_bytes({"namespace": namespace, "parts": parts})
    return hashlib.sha256(payload).hexdigest()[:length]


def freeze_json(value: Any) -> Any:
    """Recursively freeze a JSON-shaped value.

    Route messages are frozen dataclasses, but a normal ``dict`` payload would
    still be mutable.  Mapping proxies and tuples close that integrity hole.
    """

    if isinstance(value, Mapping):
        frozen = {str(k): freeze_json(v) for k, v in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((freeze_json(v) for v in value), key=repr))
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return MappingProxyType({"__bytes_sha256__": hashlib.sha256(raw).hexdigest(), "length": len(raw)})
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return freeze_json(value.to_dict())
    raise TypeError(f"value of type {type(value).__name__} is not JSON-shaped")


def thaw_json(value: Any) -> Any:
    """Convert a frozen JSON-shaped value back to mutable JSON containers."""

    if isinstance(value, Mapping):
        return {str(k): thaw_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted((_jsonable(v) for v in value), key=repr)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"__bytes_sha256__": hashlib.sha256(raw).hexdigest(), "length": len(raw)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")
