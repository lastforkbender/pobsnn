from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pobsnn.identity import stable_digest


@dataclass(frozen=True, slots=True)
class RecursiveIdentity:
    """Deterministic lineage identity for one controller treatment.

    v1.5 records lineage; it does not yet implement recursive cognition.  The
    identity is the source coordinate that later maturity layers can use
    without rewriting earlier evidence.
    """

    recursive_id: str
    run_id: str
    episode_id: str
    controller_id: str
    parent_recursive_id: str | None
    depth: int
    branch_index: int
    source_params_fingerprint: str

    @classmethod
    def root(
        cls,
        run_id: str,
        episode_id: str,
        controller_id: str,
        *,
        source_params: Any,
    ) -> "RecursiveIdentity":
        fingerprint = stable_digest("recursive-source", source_params, length=32)
        rid = "R-" + stable_digest(
            "recursive-root",
            run_id,
            episode_id,
            controller_id,
            fingerprint,
        )
        return cls(
            recursive_id=rid,
            run_id=run_id,
            episode_id=episode_id,
            controller_id=controller_id,
            parent_recursive_id=None,
            depth=0,
            branch_index=0,
            source_params_fingerprint=fingerprint,
        )

    def child(
        self,
        controller_id: str,
        *,
        branch_index: int,
        source_params: Any,
    ) -> "RecursiveIdentity":
        if branch_index < 0:
            raise ValueError("branch_index must be non-negative")
        fingerprint = stable_digest("recursive-source", source_params, length=32)
        rid = "R-" + stable_digest(
            "recursive-child",
            self.recursive_id,
            controller_id,
            branch_index,
            fingerprint,
        )
        return RecursiveIdentity(
            recursive_id=rid,
            run_id=self.run_id,
            episode_id=self.episode_id,
            controller_id=controller_id,
            parent_recursive_id=self.recursive_id,
            depth=self.depth + 1,
            branch_index=int(branch_index),
            source_params_fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
