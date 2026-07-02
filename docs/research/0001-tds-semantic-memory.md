# Research Note 0001 — TDS Semantic Memory Boundary

POBSNN computes; Staqtapp-TDS stores.

POBSNN v1 integrates TDS through `TDSVFSStore`, an adapter around the published v2.6 API:

```python
from staqtapp_tds import TDSFileSystem, TDSPersistence
```

The adapter writes JSON-compatible semantic records into TDS directories. It does not modify TDS internals, import private TDS modules, or require TDS source changes.

The semantic tree stores run history and observability artifacts rather than opaque blobs. This makes training replay, controller review, and future dashboard visualization clean.
