"""Core abstract base classes for the anomaly detection framework."""

from .base_encoder import BaseEncoder
from .base_algorithm import BaseAnomalyAlgorithm

__all__ = ['BaseEncoder', 'BaseAnomalyAlgorithm']
