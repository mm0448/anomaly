"""DataLoader utilities for creating train and test loaders."""

from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from .image_dataset import ImageDataset
from .transforms import get_standard_transforms


def create_dataloaders(
    config: Dict
) -> Tuple[DataLoader, DataLoader, ImageDataset]:
    """
    Create train and test dataloaders from configuration.

    Args:
        config: Data configuration dictionary with keys:
            - train_dir: Path to training images (normal samples)
            - test_dir_bad: Path to test images (anomalous)
            - test_dir_bad_masks: Path to ground truth masks for anomalous samples
            - test_dir_good: Path to test images (normal)
            - batch_size: Batch size for dataloaders
            - num_workers: Number of workers for data loading
            - shuffle: Whether to shuffle data

    Returns:
        Tuple of (train_loader, test_loader, test_dataset)
        test_dataset is returned separately for evaluation purposes
    """
    # Get transforms
    image_transform, mask_transform = get_standard_transforms(config)

    # Create training dataset (only normal samples)
    train_dataset = ImageDataset(
        Path(config['train_dir']),
        transform=image_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=config.get('shuffle', False),
        num_workers=config.get('num_workers', 0),
    )

    # Create test dataset (anomalous + normal samples)
    test_dataset = ImageDataset(
        Path(config['test_dir_bad']),
        transform=image_transform,
        mask_dir=Path(config['test_dir_bad_masks']),
        mask_transform=mask_transform
    )

    # Add good samples to test set
    test_dataset_good = ImageDataset(
        Path(config['test_dir_good']),
        transform=image_transform,
        mask_transform=mask_transform
    )

    # Store number of anomalous samples for later use
    test_dataset.n_anomalous = len(test_dataset)
    test_dataset.join(test_dataset_good)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
    )

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Test set: {test_dataset.n_anomalous} anomalous + "
          f"{len(test_dataset_good)} good = {len(test_dataset)} total")

    return train_loader, test_loader, test_dataset
