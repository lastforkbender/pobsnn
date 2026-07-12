# POBSNN v1.5.0 — Recursive Evidence Substrate

**POBSNN** remains a working code name for the research system currently expanded as **Policy-Orchestrated B-Spline Neural Network**.

v1.5.0 transforms the v1.4 communication scaffold into an executable, integrity-oriented process stride. The mathematical core remains a CPU-only, fixed-knot, additive B-spline regression layer. This release does not claim a general hidden-layer neural network or a mature cognitive system.

## Release purpose

v1.5 establishes the source-preserving substrate required before origamic unions, interval entropy, abstraction, prospective cognitive arrival, or mature personality can be implemented responsibly.

```text
B-spline training evidence
        ↓
meta-controller proposals
        ↓
trainer-integrated typed routing
        ↓
recursive source identities
        ↓
controller orthographic positions
        ↓
signed controller intervals
        ↓
bounded cognitive-length terms
        ↓
immutable communication episode
        ↓
POBSNN source scoring
        ↓
optional TDS v3.5.2 CSV evidence + Spiral ordering
```

## What v1.5 implements

### Mathematical hardening

- Correct multi-output MSE gradient normalization over `samples × outputs`.
- Candidate coefficient updates are validated before commit.
- Non-finite candidate updates are rejected without corrupting the layer.
- Trainer configuration and training arrays fail fast on invalid values.

### Integrated communication

- The trainer now creates typed messages, selects semantically correct controllers, executes controller responses, aggregates them, and records complete route traces.
- `MC-3` is correctly treated as the Topologic Stability Observer.
- Route message payloads are deeply immutable.
- Message, route, spawn, episode, stride, and recursive identities use stable SHA-256-derived identifiers.

### Recursive evidence substrate

- `RecursiveIdentity` preserves controller lineage and source-parameter fingerprints.
- `ControllerPosition` records transparent orthographic concern/evidence coordinates.
- `IntervalFieldSnapshot` preserves signed pairwise controller intervals.
- `CognitiveLengthStep` records dimensional movement tied to recursive identity.
- `CommunicationEpisode` closes one bounded process stride.

These records do **not** implement novelty. v1.5 explicitly prohibits a novelty score.

### Reasoning/storage separation

Trace construction, source scoring, controller profiling, advice, and policy review execute whether or not epoch persistence is enabled.

```text
POBSNN reasons first.
Storage receives completed evidence afterward.
```

### Bounded hot memory

- trainer history uses a bounded deque;
- route history uses a bounded deque;
- trace-score memory uses a bounded deque;
- policy decision history uses a bounded deque.

## TDS v3.5.2 integration

`TDSVFSStore` now targets the authoritative Staqtapp-TDS v3.5.2 public API.

It uses:

- result-first JSON reads and writes;
- validated `.tds` flush/load reopening;
- managed CSV prepare/validate/commit transactions;
- original-byte CSV preservation;
- deterministic scan and row-anchor evidence;
- content-neutral Spiral ordering of POBSNN-supplied trace metadata;
- fail-closed storage errors.

TDS records and orders supplied evidence. It does not define controller semantics, cognitive arrival, novelty, abstraction, or policy truth.

## Residual CSV families

Each persisted stride can produce four immutable source families:

```text
controller_positions
vote_intervals
cognitive_length
route_responses
```

The system writes complete CSV artifacts, not per-cell or per-row TDS hot-path mutations.

## Reserved maturity boundary

The following are represented in the release manifest but remain deliberately unimplemented:

- discrete pre-model fold-orientation boundaries;
- origamic unions and reversible manifestations;
- sine/cosine manifestation tuning;
- commutator-axis processing;
- interval entropy and abstraction formation;
- prospective cognitive arrival;
- mature developmental personality.

## Installation

Core:

```bash
python -m pip install -e .
python -m pip install -e .[test]
pytest -q
```

TDS integration uses the separately supplied Staqtapp-TDS v3.5.2 package:

```bash
python -m pip install -e /path/to/staqtapp_tds_v3_5_2
pytest -q
```

## Demo

```bash
python demos/train_v1_5_recursive_evidence.py
```

## Integrity references

The repository includes the v1.4 rigorous audit and the Ground Zero artifact-integrity manifest under `docs/reference/`. The original v1.4 archive is not embedded or modified.
