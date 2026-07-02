# POBSNN v1.3.0

**Policy-Orchestrated B-Spline Neural Network**

POBSNN is a CPU-first research platform for transparent, policy-governed adaptive B-spline neural computation. To train on material science datasets; and take full advantage of Staqtapp-TDS's vfs/directory semantics integrations + Spiral trace ranking feedbacks.

## v1.3.0 Focus

v1.3.0 adds **Trace-Ranked Controller Intelligence**.

This release keeps the hard boundary intact:

- POBSNN owns training, telemetry, trace ranking, controller advice, policy decisions, and evolution logic.
- Staqtapp-TDS remains an external library and stores serialized semantic records only.
- TDS does not reason, rank, aggregate, mutate, or participate in controller decisions.

## Current Architecture

- CPU-only NumPy B-spline neural core
- Structural telemetry
- SVD metrics
- Meta-controller observer stack
- Spawn arbiter with observation-only spawned controllers
- Policy-gated training loop
- POBSNN-owned evolution traces
- Trace ranking and controller strategy advice
- Optional TDS VFS semantic memory adapter

## New v1.3.0 Modules

```text
pobsnn/evolution/
  trace_score.py
  trace_ranker.py
  trace_memory.py

pobsnn/controllers/
  controller_profile.py
  strategy_advisor.py

pobsnn/policy/
  controller_update_policy.py
```

## Safety Rule

Trace ranking may advise controller strategy.

Trace ranking may not directly mutate controllers, network structure, or TDS.

All controller update recommendations remain advisory and policy-gated.

## Test Status

With Staqtapp-TDS v2.6.0 available on `PYTHONPATH`:

```text
19 passed
```

Without TDS installed, the optional adapter test is skipped:

```text
18 passed, 1 skipped
```

## Optional TDS Integration

POBSNN consumes the published TDS API only:

```python
from pobsnn.storage import TDSVFSStore

store = TDSVFSStore("./pobsnn_tds_memory")
```

POBSNN never modifies the Staqtapp-TDS library.
