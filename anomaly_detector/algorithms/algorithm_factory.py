"""Factory for creating anomaly detection algorithms."""

from typing import Dict
from ..core.base_algorithm import BaseAnomalyAlgorithm
from .padim import PaDiMAlgorithm


def create_algorithm(
    config: Dict,
    encoder_dims: Dict[str, int] = None
) -> BaseAnomalyAlgorithm:
    """
    Create an anomaly detection algorithm from configuration.

    Args:
        config: Algorithm configuration dictionary with keys:
            - type: Algorithm type ('padim', 'patchcore', etc.)
            - [algorithm-specific parameters]
        encoder_dims: Dictionary of feature dimensions from encoder
                     (used for auto-computing total dimensions)

    Returns:
        Initialized algorithm instance

    Raises:
        ValueError: If algorithm type is unknown
    """
    algorithm_type = config.get('type', 'padim').lower()

    if algorithm_type == 'padim':
        padim_config = config.get('padim', {})

        # Compute total dimensions from encoder if not specified
        total_dims = padim_config.get('total_dimensions')
        if total_dims is None and encoder_dims is not None:
            total_dims = sum(encoder_dims.values())

        return PaDiMAlgorithm(
            config=padim_config,
            dimension_reduction=padim_config.get('dimension_reduction', 100),
            random_seed=config.get('random_seed', 42)
        )
    else:
        raise ValueError(
            f"Unknown algorithm type: {algorithm_type}. "
            f"Supported types: ['padim']"
        )
