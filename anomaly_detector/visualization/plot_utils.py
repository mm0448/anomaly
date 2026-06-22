"""Plotting utilities for visualization."""

import numpy as np


def denormalization(x: np.ndarray) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        x: Normalized image array of shape (C, H, W)

    Returns:
        Denormalized uint8 image array of shape (H, W, C)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x
