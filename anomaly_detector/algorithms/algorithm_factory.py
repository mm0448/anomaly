"""Factory for creating anomaly detection algorithms."""

from typing import Dict
from ..core.base_algorithm import BaseAnomalyAlgorithm
from .padim import PaDiMAlgorithm
from .vae_anomaly import VAEAnomalyAlgorithm


def create_algorithm(
    config: Dict,
    encoder_dims: Dict[str, int] = None,
    device: str = 'cpu'
) -> BaseAnomalyAlgorithm:
    """
    Create an anomaly detection algorithm from configuration.

    Args:
        config: Algorithm configuration dictionary with keys:
            - type: Algorithm type ('padim', 'vae', etc.)
            - [algorithm-specific parameters]
        encoder_dims: Dictionary of feature dimensions from encoder
                     (used for auto-computing total dimensions)
        device: Device to run algorithm on

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

    elif algorithm_type == 'vae':
        vae_config = config.get('vae', {})

        return VAEAnomalyAlgorithm(
            config=vae_config,
            dim_h=vae_config.get('dim_h', 256),
            dim_z=vae_config.get('dim_z', 64),
            learning_rate=vae_config.get('learning_rate', 1e-3),
            num_epochs=vae_config.get('num_epochs', 50),
            batch_size=vae_config.get('batch_size', 32),
            beta=vae_config.get('beta', 1.0),
            validation_split=vae_config.get('validation_split', 0.2),
            random_seed=config.get('random_seed', 42),
            device=device
        )

    else:
        raise ValueError(
            f"Unknown algorithm type: {algorithm_type}. "
            f"Supported types: ['padim', 'vae']"
        )
