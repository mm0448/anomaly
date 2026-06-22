"""Random seed utilities for reproducibility."""

import random
import numpy as np
import torch


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    This ensures reproducible results across runs.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
