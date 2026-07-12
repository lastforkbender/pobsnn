from __future__ import annotations

import json
from typing import Any


POSITION_FIELDS = (
    "run_id", "episode_id", "stride_id", "epoch", "controller_id", "recursive_id",
    "axis_id", "position", "confidence", "source_titles_json",
)
INTERVAL_FIELDS = (
    "run_id", "episode_id", "stride_id", "epoch", "controller_a", "controller_b",
    "recursive_id_a", "recursive_id_b", "axis_id", "signed_interval", "l2_distance",
)
LENGTH_FIELDS = (
    "run_id", "episode_id", "stride_id", "epoch", "controller_id", "recursive_id",
    "axis_id", "component_delta", "step_length", "cumulative_length",
)
ROUTE_FIELDS = (
    "run_id", "episode_id", "stride_id", "epoch", "route_trace_id", "message_id",
    "message_type", "source", "severity", "target_controller", "recursive_id",
    "accepted", "confidence", "aggregate_action",
)


def episode_csv_families(episode: dict[str, Any]) -> dict[str, tuple[tuple[str, ...], list[dict[str, Any]]]]:
    run_id = str(episode["run_id"])
    episode_id = str(episode["episode_id"])
    stride_id = str(episode["stride_id"])
    epoch = int(episode["epoch"])
    interval = episode.get("interval_field", {})
    axes = tuple(str(v) for v in interval.get("axes", []))

    positions: list[dict[str, Any]] = []
    for position in interval.get("positions", []):
        values = {str(item["axis_id"]): float(item["value"]) for item in position.get("coordinates", [])}
        for axis in axes:
            positions.append({
                "run_id": run_id,
                "episode_id": episode_id,
                "stride_id": stride_id,
                "epoch": epoch,
                "controller_id": position["controller_id"],
                "recursive_id": position["recursive_id"],
                "axis_id": axis,
                "position": values.get(axis, 0.0),
                "confidence": float(position.get("confidence", 0.0)),
                "source_titles_json": json.dumps(position.get("source_titles", []), separators=(",", ":"), ensure_ascii=False),
            })

    intervals: list[dict[str, Any]] = []
    for item in interval.get("intervals", []):
        item_axes = tuple(str(v) for v in item.get("axes", axes))
        for axis, delta in zip(item_axes, item.get("axis_deltas", [])):
            intervals.append({
                "run_id": run_id,
                "episode_id": episode_id,
                "stride_id": stride_id,
                "epoch": epoch,
                "controller_a": item["controller_a"],
                "controller_b": item["controller_b"],
                "recursive_id_a": item["recursive_id_a"],
                "recursive_id_b": item["recursive_id_b"],
                "axis_id": axis,
                "signed_interval": float(delta),
                "l2_distance": float(item.get("l2_distance", 0.0)),
            })

    lengths: list[dict[str, Any]] = []
    for item in episode.get("cognitive_length", []):
        item_axes = tuple(str(v) for v in item.get("axes", axes))
        for axis, delta in zip(item_axes, item.get("component_deltas", [])):
            lengths.append({
                "run_id": run_id,
                "episode_id": episode_id,
                "stride_id": stride_id,
                "epoch": epoch,
                "controller_id": item["controller_id"],
                "recursive_id": item["recursive_id"],
                "axis_id": axis,
                "component_delta": float(delta),
                "step_length": float(item.get("step_length", 0.0)),
                "cumulative_length": float(item.get("cumulative_length", 0.0)),
            })

    routes: list[dict[str, Any]] = []
    for trace in episode.get("route_traces", []):
        message = trace.get("message", {})
        aggregate = trace.get("aggregate", {})
        response_by_controller = {r["controller_id"]: r for r in trace.get("responses", [])}
        for target in trace.get("targets", []):
            response = response_by_controller.get(target, {})
            routes.append({
                "run_id": run_id,
                "episode_id": episode_id,
                "stride_id": stride_id,
                "epoch": epoch,
                "route_trace_id": trace.get("trace_id", ""),
                "message_id": message.get("message_id", ""),
                "message_type": message.get("message_type", ""),
                "source": message.get("source", ""),
                "severity": float(message.get("severity", 0.0)),
                "target_controller": target,
                "recursive_id": response.get("recursive_id", ""),
                "accepted": bool(response.get("accepted", False)),
                "confidence": float(response.get("confidence", 0.0)),
                "aggregate_action": aggregate.get("action", ""),
            })

    return {
        "controller_positions": (POSITION_FIELDS, positions),
        "vote_intervals": (INTERVAL_FIELDS, intervals),
        "cognitive_length": (LENGTH_FIELDS, lengths),
        "route_responses": (ROUTE_FIELDS, routes),
    }
