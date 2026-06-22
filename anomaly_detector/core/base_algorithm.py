"""Abstract base class for anomaly detection algorithms."""

from abc import ABC, abstractmethod
from typing import Dict
import torch
import numpy as np


class BaseAnomalyAlgorithm(ABC):
    """
    Abstract base class for anomaly detection algorithms.

    Algorithms learn a representation of normal data during training,
    then compute anomaly scores for test samples. Different algorithms
    use different approaches (Gaussian modeling, memory banks, flows, etc.).
    """

    def __init__(self, config: Dict):
        """
        Initialize algorithm.

        Args:
            config: Algorithm-specific configuration dictionary
        """
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(self, features: Dict[str, torch.Tensor]) -> None:
        """
        Fit the anomaly detection model on normal training features.

        This method learns the distribution/representation of normal data.

        Args:
            features: Dictionary mapping layer names to feature tensors
                     Each tensor has shape (N, C, H, W)
        """
        pass

    @abstractmethod
    def predict(self, features: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute anomaly scores for test features.

        Higher scores indicate higher likelihood of anomaly.

        Args:
            features: Dictionary mapping layer names to feature tensors
                     Each tensor has shape (N, C, H, W)

        Returns:
            Anomaly score maps of shape (N, H, W) with values in [0, 1]
            where H, W match the original image resolution
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save fitted model parameters to disk.

        Args:
            path: File path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load fitted model parameters from disk.

        Args:
            path: File path to load model from
        """
        pass
