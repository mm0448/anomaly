"""Feature caching utilities to avoid redundant computation."""

import pickle
from pathlib import Path
from typing import Dict, Optional
import torch


class FeatureCache:
    """
    Manages caching of extracted features to disk.

    Features are expensive to extract, so caching them speeds up
    experimentation with different algorithms.
    """

    def __init__(self, cache_config: Dict):
        """
        Initialize feature cache.

        Args:
            cache_config: Cache configuration dictionary with keys:
                - enabled: Whether caching is enabled
                - cache_dir: Directory to store cache files
                - train_features_file: Filename for training features
                - use_cached_if_exists: Whether to use existing cache
        """
        self.enabled = cache_config.get('enabled', True)
        self.cache_dir = Path(cache_config.get('cache_dir', 'outputs/cache'))
        self.train_features_file = cache_config.get(
            'train_features_file', 'train_features.pkl'
        )
        self.use_cached = cache_config.get('use_cached_if_exists', True)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_train_features_path(self) -> Path:
        """Get path to training features cache file."""
        return self.cache_dir / self.train_features_file

    def has_cached_train_features(self) -> bool:
        """
        Check if cached training features exist.

        Returns:
            True if cache exists and should be used
        """
        if not self.enabled or not self.use_cached:
            return False

        return self._get_train_features_path().exists()

    def save_train_features(self, features: Dict[str, torch.Tensor]) -> None:
        """
        Save training features to cache.

        Args:
            features: Dictionary mapping layer names to feature tensors
        """
        if not self.enabled:
            return

        cache_path = self._get_train_features_path()
        print(f"Saving training features to cache: {cache_path}")

        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

    def load_train_features(self) -> Dict[str, torch.Tensor]:
        """
        Load training features from cache.

        Returns:
            Dictionary mapping layer names to feature tensors

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = self._get_train_features_path()

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        print(f"Loading training features from cache: {cache_path}")

        with open(cache_path, 'rb') as f:
            features = pickle.load(f)

        return features

    def clear_cache(self) -> None:
        """Remove all cache files."""
        if self.enabled and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            print(f"Cleared cache directory: {self.cache_dir}")
