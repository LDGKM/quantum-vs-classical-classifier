"""quantum-vs-classical-classifier : comparer des classifieurs quantiques et classiques."""

from .data import load_dataset, DATASETS
from .classical import classical_models
from .quantum import VariationalQuantumClassifier, QuantumKernelSVC
from .benchmark import run_benchmark

__all__ = [
    "load_dataset",
    "DATASETS",
    "classical_models",
    "VariationalQuantumClassifier",
    "QuantumKernelSVC",
    "run_benchmark",
]

__version__ = "0.1.0"
