# Research Note 0002 — Phase 3 Policy-Gated Training

The Phase 3 loop is:

```text
forward pass
→ loss
→ coefficient update
→ structural telemetry
→ SVD summary
→ compression state
→ typed node classification
→ recursion gate
→ controller observations
→ policy decisions
→ TDS semantic persistence
```

Only the optimizer updates spline coefficients. Structural mutation remains disabled in v1.

Observer spawning is permitted, but spawned controllers are observation-only and expire after a small number of observations.
