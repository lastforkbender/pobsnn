from __future__ import annotations

from pobsnn.routing import AggregationBus, ControllerRouteResponse, ControllerRouter, make_message
from pobsnn.storage import MemoryStore


def main() -> None:
    store = MemoryStore()
    router = ControllerRouter(store=store, store_path="/routing/demo")
    message = make_message(
        "CompressionAlert",
        "svd_metrics",
        severity=0.91,
        epoch=12,
        payload={"compression_pressure": 0.84, "effective_rank": 6},
        tags=("cpu", "policy-orchestrated"),
    )
    targets = router.targets_for(message)
    responses = [
        ControllerRouteResponse("MC-1", True, 0.71, "utilization drop confirmed"),
        ControllerRouteResponse("MC-3", True, 0.93, "compression pressure high"),
        ControllerRouteResponse("MC-4", True, 0.82, "stable enough for policy review"),
        ControllerRouteResponse("MC-5", False, 0.44, "spawn deferred"),
    ]
    final_trace = AggregationBus().aggregate(message, targets, responses)
    router.record(final_trace)
    print(final_trace.to_dict())


if __name__ == "__main__":
    main()
