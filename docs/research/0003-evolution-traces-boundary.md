# 0003 — Evolution Traces and the TDS Boundary

POBSNN v1.2.0 introduces `EvolutionTrace` as a first-class internal record of structural learning.

An Evolution Trace may include loss, score delta, SVD state, compression pressure, recursion gate output, controller proposals, and policy decisions.

The trace belongs to POBSNN. Storage backends receive the serialized trace only.

This prevents TDS from becoming a reasoning layer. TDS is used for deterministic semantic persistence, not for deciding whether a trace is good, bad, useful, redundant, or actionable.

Design rule:

```text
POBSNN owns interpretation.
TDS owns persistence.
```
