"""POBSNN v1.5: recursive evidence substrate for CPU B-spline research."""

from pobsnn.core import BSplineBasis, BSplineBasisConfig, BSplineLayer, BSplineNeuron
from pobsnn.controllers import MetaControllerStack
from pobsnn.development import (
    CognitiveLengthStep,
    CommunicationEpisode,
    ControllerPosition,
    IntervalFieldSnapshot,
    RecursiveIdentity,
)
from pobsnn.evolution import EvolutionTrace
from pobsnn.storage import MemoryStore, TDSVFSStore
from pobsnn.training import PolicyGatedTrainer, TrainerConfig

__version__ = "1.5.0"

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
    "RecursiveIdentity",
    "ControllerPosition",
    "IntervalFieldSnapshot",
    "CognitiveLengthStep",
    "CommunicationEpisode",
]
