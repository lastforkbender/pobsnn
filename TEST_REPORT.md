# POBSNN v1.5.0 Validation Report

## Release

POBSNN v1.5.0 — Recursive Evidence Substrate

## Environment

- Python: 3.13.5
- NumPy: 2.3.5
- pytest: 9.0.2
- Optional integration: Staqtapp-TDS v3.5.2 source package
- Execution: CPU only
- GPU/CUDA: not used

## Core-only test run

```bash
PYTHONPATH=. pytest -q
```

```text
36 passed, 2 skipped
```

The two skips are TDS-dependent tests.

## TDS v3.5.2 integration run

```bash
PYTHONPATH=.:/path/to/staqtapp_tds_v3_5_2/src pytest -q
```

```text
38 passed
```

## Coverage

```bash
PYTHONPATH=.:/path/to/staqtapp_tds_v3_5_2/src \
pytest --cov=pobsnn --cov-report=term-missing -q
```

```text
1,803 statements
195 missed
89% total coverage
38 passed
```

The new development modules have the following observed statement coverage:

- cognitive-length tracker: 100%
- communication episode: 100%
- interval field: 100%
- controller position substrate: 98%
- recursive identity: 81%
- residual CSV shaping: 100%

## Build verification

```bash
python -m pip wheel . --no-deps -w dist
```

A `pobsnn-1.5.0-py3-none-any.whl` wheel built successfully and imported from an isolated target directory.

## Demo verification

All six included demos exited successfully:

- route meta-controllers
- base B-spline regression
- Phase 1 observers
- Phase 2 spawning
- v1 TDS semantic-memory demo in memory mode
- v1.5 recursive-evidence demo

## TDS end-to-end smoke run

A two-epoch trainer run using `TDSVFSStore` verified:

- result-first JSON persistence;
- four managed CSV residual families per stride;
- CSV transaction validation and commit;
- scan and row-anchor evidence;
- TDS Spiral ordering of POBSNN source scores;
- `.tds` flush;
- reopen and read of manifest and communication episode.

## High-value hardening tests

- three-output MSE gradient matches finite differences;
- non-finite candidate update is rejected without coefficient mutation;
- invalid trainer configuration fails immediately;
- route payloads are deeply immutable;
- route IDs incorporate payload and event identity;
- spawned-controller IDs are stable across `PYTHONHASHSEED` values;
- trace reasoning continues with epoch persistence disabled;
- trainer, router, policy, and trace hot histories are bounded;
- communication episodes contain recursive identities, real route responses, positions, intervals, and process-length terms;
- active trace scores contain no subjective novelty field;
- TDS v3.5.2 reopen, managed CSV, and Spiral integration pass.

## Remaining research boundaries

The validation does not claim that v1.5 implements:

- general feature-interaction learning;
- hidden KnotNet layers;
- mature SVD compression;
- origamic unions;
- sine/cosine manifestation;
- interval entropy or abstraction;
- prospective cognitive arrival;
- human cognition.
