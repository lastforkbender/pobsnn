"""POBSNN v1: CPU B-spline NN with TDS-backed semantic memory adapter."""

from pobsnn.core import BSplineBasis, BSplineBasisConfig, BSplineLayer, BSplineNeuron
from pobsnn.controllers import MetaControllerStack
from pobsnn.training import PolicyGatedTrainer, TrainerConfig
from pobsnn.storage import MemoryStore, TDSVFSStore
from pobsnn.evolution import EvolutionTrace

__version__ = "1.3.0"

__all__ = [
    "BSplineBasis",
    "BSplineBasisConfig",
    "BSplineLayer",
    "BSplineNeuron",
    "MetaControllerStack",
    "PolicyGatedTrainer",
    "TrainerConfig",
    "MemoryStore",
    "TDSVFSStore",
    "EvolutionTrace",
]
