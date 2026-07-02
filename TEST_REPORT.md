# POBSNN v1.3.0 Test Report

## Environment

- Python project: POBSNN v1.3.0
- Engine: CPU-only NumPy
- Optional persistence backend: Staqtapp-TDS v2.6.0 VFS

## Results

### Full test with TDS v2.6.0 on PYTHONPATH

```text
19 passed in 1.70s
```

### Test without TDS installed

```text
18 passed, 1 skipped in 0.28s
```

The skipped test is the optional TDS adapter test. This is expected when the external Staqtapp-TDS library is unavailable.

## Coverage Added in v1.3.0

- Trace ranking scores deterministic EvolutionTrace objects.
- Trace scores remain inside POBSNN reasoning/evolution region.
- Controller profiles can learn from ranked trace evidence.
- StrategyAdvisor generates advisory-only recommendations.
- ControllerUpdatePolicy gates controller update recommendations.
- Training loop persists trace scores, strategy advice, update policy decisions, and trace memory summaries.
- TDS adapter continues to use the published TDS API without modifying TDS.

## Boundary Verified

POBSNN owns:

- trace ranking
- controller advice
- policy decisions
- evolution trace construction

TDS owns:

- storage
- VFS persistence
- serialized record retention

TDS does not reason, rank, aggregate, or mutate.
