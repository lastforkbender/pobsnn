from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pobsnn.policy import PolicyProposal


@dataclass(frozen=True)
class ControllerContext:
    structural: object
    svd: object | None = None


class MetaController(ABC):
    controller_id: str = "MC-X"
    name: str = "abstract"

    @abstractmethod
    def observe(self, context: ControllerContext) -> list[PolicyProposal]:
        """Return observation-only policy proposals."""
