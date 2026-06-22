"""PaDiM: Patch Distribution Modeling for anomaly detection."""

import pickle
from pathlib import Path
from random import sample
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

from ..core.base_algorithm import BaseAnomalyAlgorithm


class PaDiMAlgorithm(BaseAnomalyAlgorithm):
    """
    PaDiM: Patch Distribution Modeling for anomaly detection.

    Fits multivariate Gaussian distributions per spatial location
    and uses Mahalanobis distance for anomaly scoring.

    Reference: https://arxiv.org/abs/2011.08785
    """

    def __init__(
        self,
        config: Dict,
        dimension_reduction: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize PaDiM algorithm.

        Args:
            config: Algorithm configuration dictionary
            dimension_reduction: Target dimensions after reduction (None = no reduction)
            random_seed: Random seed for dimension selection
        """
        super().__init__(config)
        self.dimension_reduction = dimension_reduction
        self.random_seed = random_seed
        self.mean = None
        self.cov = None
        self.idx = None  # Indices for dimension reduction
        self.gaussian_smoothing_sigma = config.get('gaussian_smoothing_sigma', 4)
        self.cov_reg = config.get('covariance_regularization', 0.01)
        self.spatial_shape = None

    def _embedding_concat(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate embeddings from two feature maps at different resolutions.

        This function spatially aligns features from different layers by:
        1. Unfolding the higher-resolution feature map into patches
        2. Concatenating with the lower-resolution features
        3. Folding back to the original spatial dimensions

        Args:
            x: Higher resolution feature map (B, C1, H1, W1)
            y: Lower resolution feature map (B, C2, H2, W2)

        Returns:
            Concatenated features with shape (B, C1+C2, H1, W1)
        """
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()

        # Calculate stride for spatial alignment
        s = int(H1 / H2)

        # Unfold x into patches matching y's spatial resolution
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)

        # Concatenate x patches with y features
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)

        # Fold back to original spatial resolution
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def _concat_multiscale_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenate multi-scale features with spatial alignment.

        Args:
            features: Dictionary mapping layer names to feature tensors

        Returns:
            Concatenated multi-scale feature tensor
        """
        layer_names = sorted(features.keys())
        embedding = features[layer_names[0]]

        for layer_name in layer_names[1:]:
            embedding = self._embedding_concat(embedding, features[layer_name])

        return embedding

    def fit(self, features: Dict[str, torch.Tensor]) -> None:
        """
        Fit Gaussian distributions on training features.

        Args:
            features: Dictionary mapping layer names to feature tensors
                     Each tensor has shape (N, C, H, W)
        """
        print("Concatenating multi-scale features...")
        embedding_vectors = self._concat_multiscale_features(features)

        b, c, h, w = embedding_vectors.size()

        # Dimension reduction
        if self.dimension_reduction and self.dimension_reduction < c:
            print(f"Reducing dimensions from {c} to {self.dimension_reduction}...")
            # Use random.sample() to match original implementation
            self.idx = torch.tensor(sample(range(0, c), self.dimension_reduction))
            embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)
            c = self.dimension_reduction

        # Reshape for per-location processing
        embedding_vectors = embedding_vectors.view(b, c, h * w)

        # Compute mean and covariance per spatial location
        print("Fitting Gaussian distributions...")
        self.mean = torch.mean(embedding_vectors, dim=0).numpy()  # Shape: (C, H*W)
        self.cov = torch.zeros(c, c, h * w).numpy()  # Shape: (C, C, H*W)

        for i in range(h * w):
            # Compute covariance matrix with regularization
            cov_matrix = np.cov(
                embedding_vectors[:, :, i].numpy(),
                rowvar=False
            )
            # Add regularization for numerical stability
            self.cov[:, :, i] = cov_matrix + self.cov_reg * np.identity(c)

        self.is_fitted = True
        self.spatial_shape = (h, w)

    def predict(self, features: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute Mahalanobis distance anomaly scores.

        Args:
            features: Dictionary mapping layer names to feature tensors
                     Each tensor has shape (N, C, H, W)

        Returns:
            Anomaly score maps of shape (N, H, W) with values in [0, 1]

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Multi-scale feature concatenation
        embedding_vectors = self._concat_multiscale_features(features)

        # Apply dimension reduction
        if self.idx is not None:
            embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # Compute Mahalanobis distance
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

        print("Computing Mahalanobis distances...")
        dist_list = []
        for i in range(H * W):
            mean = self.mean[:, i]
            cov_inv = np.linalg.inv(self.cov[:, :, i])

            dist = [
                mahalanobis(sample[:, i], mean, cov_inv)
                for sample in embedding_vectors
            ]
            dist_list.append(dist)

        # Reshape to spatial maps
        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # Upsample to original image resolution (224x224)
        dist_tensor = torch.tensor(dist_list)
        score_map = F.interpolate(
            dist_tensor.unsqueeze(1),
            size=224,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Apply Gaussian smoothing
        print("Smoothing anomaly maps...")
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(
                score_map[i],
                sigma=self.gaussian_smoothing_sigma
            )

        # Normalize to [0, 1]
        max_score = score_map.max()
        min_score = score_map.min()
        if max_score > min_score:
            scores = (score_map - min_score) / (max_score - min_score)
        else:
            scores = score_map

        return scores

    def save(self, path: str) -> None:
        """
        Save fitted parameters to disk.

        Args:
            path: File path to save model
        """
        save_data = {
            'mean': self.mean,
            'cov': self.cov,
            'idx': self.idx,
            'spatial_shape': self.spatial_shape,
            'config': self.config
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, path: str) -> None:
        """
        Load fitted parameters from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.mean = save_data['mean']
        self.cov = save_data['cov']
        self.idx = save_data.get('idx')
        self.spatial_shape = save_data['spatial_shape']
        self.is_fitted = True
