"""Factory for creating feature encoders."""

from typing import Dict
from ..core.base_encoder import BaseEncoder
from .resnet_encoder import ResNetEncoder


def create_encoder(config: Dict, device: str = 'cpu') -> BaseEncoder:
    """
    Create a feature encoder from configuration.

    Args:
        config: Encoder configuration dictionary with keys:
            - type: Encoder type ('resnet', 'efficientnet', etc.)
            - backbone: Specific backbone variant
            - layers: List of layers to extract features from
            - pretrained: Whether to use pretrained weights
        device: Device to run encoder on

    Returns:
        Initialized encoder instance

    Raises:
        ValueError: If encoder type is unknown
    """
    encoder_type = config.get('type', 'resnet').lower()

    if encoder_type == 'resnet':
        return ResNetEncoder(
            backbone=config.get('backbone', 'resnet18'),
            layers=config.get('layers', ['layer1', 'layer2', 'layer3']),
            pretrained=config.get('pretrained', True),
            device=device
        )
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Supported types: ['resnet']"
        )
