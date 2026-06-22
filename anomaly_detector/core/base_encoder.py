"""Abstract base class for feature encoders."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class BaseEncoder(ABC):
    """
    Abstract base class for feature encoders.

    Encoders extract multi-scale features from images using neural network
    backbones (e.g., ResNet, EfficientNet). Features are typically extracted
    from multiple intermediate layers to capture both low-level and high-level
    representations.
    """

    def __init__(self, layers: List[str], device: str = 'cpu'):
        """
        Initialize encoder.

        Args:
            layers: List of layer names to extract features from
            device: Device to run model on ('cpu' or 'cuda:0')
        """
        self.layers = layers
        self.device = device
        self.model: nn.Module = None
        self.hooks: List = []

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Build and return the encoder model.

        Returns:
            Initialized neural network model
        """
        pass

    @abstractmethod
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get feature dimensions for each layer.

        Returns:
            Dictionary mapping layer names to feature dimensions
            Example: {'layer1': 64, 'layer2': 128, 'layer3': 256}
        """
        pass

    @abstractmethod
    def extract_features(self, dataloader) -> Dict[str, torch.Tensor]:
        """
        Extract features from all images in a dataloader.

        Args:
            dataloader: PyTorch DataLoader yielding (image, mask) tuples

        Returns:
            Dictionary mapping layer names to feature tensors
            Each tensor has shape (N, C, H, W) where:
                N = number of samples
                C = feature channels
                H, W = spatial dimensions
        """
        pass

    def cleanup(self):
        """Remove forward hooks and cleanup resources."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks when encoder is deleted."""
        self.cleanup()
