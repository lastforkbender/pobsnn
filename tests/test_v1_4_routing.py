from __future__ import annotations

from pobsnn.routing import (
    AggregationBus,
    ControllerRouteResponse,
    ControllerRouter,
    RoutePolicy,
    make_message,
)
from pobsnn.storage import MemoryStore


def test_route_policy_maps_compression_to_expected_controllers() -> None:
    msg = make_message("CompressionAlert", "telemetry", severity=0.9, epoch=7)
    trace = ControllerRouter().route(msg)
    assert trace.targets == ("MC-1", "MC-3", "MC-4", "MC-5")
    assert trace.trace_id.startswith("route:")


def test_unknown_route_uses_actual_stability_controller() -> None:
    policy = RoutePolicy.default()
    msg = make_message("UnknownDiagnostic", "telemetry")
    assert policy.targets_for(msg) == ("MC-3",)


def test_aggregation_bus_escalates_severe_accepted_route() -> None:
    msg = make_message("NodeInstabilityAlert", "telemetry", severity=0.95, node_id=3)
    targets = ("MC-3", "MC-5")
    responses = [
        ControllerRouteResponse("MC-3", True, 0.88, "stability risk confirmed"),
        ControllerRouteResponse("MC-5", False, 0.35, "spawn not required"),
    ]
    aggregate = AggregationBus(min_confidence=0.5).aggregate(msg, targets, responses)
    assert aggregate.aggregate["action"] == "escalate_to_policy"
    assert aggregate.aggregate["accepted_response_count"] == 1
    assert aggregate.responses[0].controller_id == "MC-3"


def test_router_persists_route_traces_to_memory_store() -> None:
    store = MemoryStore()
    router = ControllerRouter(store=store, store_path="/routing/events")
    trace = router.route(make_message("TraceRankingUpdate", "trace_ranker", severity=0.4, epoch=10))
    saved = store.read_json("/routing/events", trace.trace_id.replace(":", "_"))
    assert saved["message"]["message_type"] == "TraceRankingUpdate"
    assert saved["targets"] == ["MC-0", "MC-3", "MC-5"]


def test_router_history_is_bounded() -> None:
    router = ControllerRouter(max_history=2)
    router.route(make_message("CompressionAlert", "t", epoch=1))
    router.route(make_message("CompressionAlert", "t", epoch=2))
    router.route(make_message("CompressionAlert", "t", epoch=3))
    assert len(router.history) == 2
    assert router.history[0].message.epoch == 2
