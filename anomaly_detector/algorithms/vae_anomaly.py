"""VAE-based anomaly detection using reconstruction error."""

import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from ..core.base_algorithm import BaseAnomalyAlgorithm
from .vae import SimpleVAE


class VAEAnomalyAlgorithm(BaseAnomalyAlgorithm):
    """
    VAE-based anomaly detection.

    Trains a VAE to reconstruct normal features. Anomalies are detected
    by measuring reconstruction error - anomalous samples should have
    higher reconstruction error.
    """

    def __init__(
        self,
        config: Dict,
        dim_h: int = 256,
        dim_z: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        batch_size: int = 32,
        beta: float = 1.0,
        validation_split: float = 0.2,
        random_seed: int = 42,
        device: str = 'cpu'
    ):
        """
        Initialize VAE anomaly detection algorithm.

        Args:
            config: Algorithm configuration dictionary
            dim_h: Hidden layer dimensions
            dim_z: Latent space dimensions
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            beta: Weight for KL divergence term (beta-VAE)
            validation_split: Fraction of data to use for validation (0.0-1.0)
            random_seed: Random seed
            device: Device to train on
        """
        super().__init__(config)
        self.dim_h = dim_h
        self.dim_z = dim_z
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.device = device
        self.vae = None
        self.feature_dim = None
        self.spatial_shape = None

        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _concat_multiscale_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenate multi-scale features along channel dimension.

        Args:
            features: Dictionary mapping layer names to feature tensors

        Returns:
            Concatenated features
        """
        layer_names = sorted(features.keys())

        # Get spatial size of smallest feature map
        min_h = min([features[k].shape[2] for k in layer_names])
        min_w = min([features[k].shape[3] for k in layer_names])

        # Resize all features to same spatial size
        resized_features = []
        for layer_name in layer_names:
            feat = features[layer_name]
            if feat.shape[2] != min_h or feat.shape[3] != min_w:
                feat = F.interpolate(
                    feat,
                    size=(min_h, min_w),
                    mode='bilinear',
                    align_corners=False
                )
            resized_features.append(feat)

        # Concatenate along channel dimension
        return torch.cat(resized_features, dim=1)

    def _vae_loss(
        self,
        x: torch.Tensor,
        x_reconst: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original input
            x_reconst: Reconstructed input
            z_mean: Latent mean
            z_logvar: Latent log variance

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_reconst, x, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def fit(self, features: Dict[str, torch.Tensor]) -> None:
        """
        Train VAE on normal features.

        Args:
            features: Dictionary mapping layer names to feature tensors
                     Each tensor has shape (N, C, H, W)
        """
        print("Preparing features for VAE training...")

        # Concatenate multi-scale features
        embedding_vectors = self._concat_multiscale_features(features)
        B, C, H, W = embedding_vectors.size()

        self.feature_dim = C
        self.spatial_shape = (H, W)

        # Split images into train and validation BEFORE flattening
        # This ensures no spatial features from validation images leak into training
        num_images = B
        val_images = int(num_images * self.validation_split)
        train_images = num_images - val_images

        # Split image indices
        indices = torch.randperm(num_images)
        train_img_indices = indices[:train_images]
        val_img_indices = indices[train_images:]

        # Split the embedding vectors by image
        train_embeddings = embedding_vectors[train_img_indices]  # (train_images, C, H, W)
        val_embeddings = embedding_vectors[val_img_indices]      # (val_images, C, H, W)

        # Now flatten each set separately
        train_data = train_embeddings.permute(0, 2, 3, 1).reshape(-1, C).to(self.device)
        val_data = val_embeddings.permute(0, 2, 3, 1).reshape(-1, C).to(self.device)

        print(f"Image split: {train_images} train images, {val_images} val images")
        print(f"Feature samples: {train_data.shape[0]} train, {val_data.shape[0]} val")

        # Initialize VAE
        self.vae = SimpleVAE(
            dim_x=C,
            dim_h=self.dim_h,
            dim_z=self.dim_z
        ).to(self.device)

        # Optimizer
        optimizer = Adam(self.vae.parameters(), lr=self.learning_rate)

        # Training loop
        print(f"Training VAE for {self.num_epochs} epochs...")
        best_val_loss = float('inf')
        train_size = train_data.shape[0]
        val_size = val_data.shape[0]

        for epoch in range(self.num_epochs):
            # Training phase
            self.vae.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            num_train_batches = 0

            # Shuffle training samples (spatial locations within train images)
            train_perm = torch.randperm(train_size)

            # Mini-batch training
            for i in range(0, train_size, self.batch_size):
                batch_indices = train_perm[i:i + self.batch_size]
                batch = train_data[batch_indices]

                # Forward pass
                x_reconst, z_mean, z_logvar = self.vae(batch)

                # Compute loss
                loss, recon_loss, kl_loss = self._vae_loss(
                    batch, x_reconst, z_mean, z_logvar
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                num_train_batches += 1

            # Validation phase
            self.vae.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for i in range(0, val_size, self.batch_size):
                    batch = val_data[i:i + self.batch_size]

                    # Forward pass
                    x_reconst, z_mean, z_logvar = self.vae(batch)

                    # Compute loss
                    loss, recon_loss, kl_loss = self._vae_loss(
                        batch, x_reconst, z_mean, z_logvar
                    )

                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    num_val_batches += 1

            # Compute average losses
            avg_train_loss = train_loss / num_train_batches
            avg_val_loss = val_loss / num_val_batches

            # Track best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_train_recon = train_recon_loss / num_train_batches
                avg_train_kl = train_kl_loss / num_train_batches
                avg_val_recon = val_recon_loss / num_val_batches
                avg_val_kl = val_kl_loss / num_val_batches

                print(f"Epoch {epoch+1}/{self.num_epochs}:")
                print(f"  Train - Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}")
                print(f"  Val   - Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}")

        self.vae.eval()
        self.is_fitted = True
        print("VAE training complete!")

    def predict(self, features: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute anomaly scores using reconstruction error.

        Args:
            features: Dictionary mapping layer names to feature tensors
                     Each tensor has shape (N, C, H, W)

        Returns:
            Anomaly score maps of shape (N, 224, 224) with values in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        print("Computing reconstruction errors...")

        # Concatenate multi-scale features
        embedding_vectors = self._concat_multiscale_features(features)
        B, C, H, W = embedding_vectors.size()

        # Reshape to (B*H*W, C)
        embedding_flat = embedding_vectors.permute(0, 2, 3, 1).reshape(-1, C)
        embedding_flat = embedding_flat.to(self.device)

        # Compute reconstruction error in batches
        self.vae.eval()
        recon_errors = []

        with torch.no_grad():
            for i in range(0, embedding_flat.shape[0], self.batch_size):
                batch = embedding_flat[i:i + self.batch_size]
                x_reconst, _, _ = self.vae(batch)

                # Compute per-sample reconstruction error (MSE)
                error = torch.mean((batch - x_reconst) ** 2, dim=1)
                recon_errors.append(error.cpu())

        # Concatenate all errors
        recon_errors = torch.cat(recon_errors, dim=0)

        # Reshape back to spatial map (B, H, W)
        recon_errors = recon_errors.reshape(B, H, W).numpy()

        # Upsample to original image resolution (224x224)
        recon_errors_tensor = torch.tensor(recon_errors).unsqueeze(1)
        score_map = F.interpolate(
            recon_errors_tensor,
            size=224,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

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
        Save fitted model to disk.

        Args:
            path: File path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'vae_state_dict': self.vae.state_dict() if self.vae else None,
            'feature_dim': self.feature_dim,
            'spatial_shape': self.spatial_shape,
            'dim_h': self.dim_h,
            'dim_z': self.dim_z,
            'config': self.config
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, path: str) -> None:
        """
        Load fitted model from disk.

        Args:
            path: File path to load model from
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.feature_dim = save_data['feature_dim']
        self.spatial_shape = save_data['spatial_shape']
        self.dim_h = save_data['dim_h']
        self.dim_z = save_data['dim_z']

        # Reconstruct VAE
        if save_data['vae_state_dict']:
            self.vae = SimpleVAE(
                dim_x=self.feature_dim,
                dim_h=self.dim_h,
                dim_z=self.dim_z
            ).to(self.device)
            self.vae.load_state_dict(save_data['vae_state_dict'])
            self.vae.eval()

        self.is_fitted = True
