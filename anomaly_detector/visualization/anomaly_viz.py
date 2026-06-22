"""Visualization utilities for anomaly detection results."""

import os
from typing import Dict, List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.segmentation import mark_boundaries

from .plot_utils import denormalization


class AnomalyVisualizer:
    """
    Visualizer for anomaly detection results.

    Creates comprehensive visualizations including:
    - ROC curves
    - Anomaly heatmaps
    - Segmentation results
    """

    def __init__(self, config: Dict):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration dictionary with keys:
                - enabled: Whether visualization is enabled
                - save_roc_curves: Whether to save ROC curves
                - save_heatmaps: Whether to save heatmaps
                - max_samples: Maximum number of samples to visualize
                - morphology_kernel_size: Kernel size for morphological operations
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.save_roc = config.get('save_roc_curves', True)
        self.save_heatmaps = config.get('save_heatmaps', True)
        self.max_samples = config.get('max_samples', None)
        self.kernel_size = config.get('morphology_kernel_size', 4)

    def plot_roc_curves(
        self,
        gt_labels: np.ndarray,
        img_scores: np.ndarray,
        gt_masks: np.ndarray,
        score_maps: np.ndarray,
        image_roc_auc: float,
        pixel_roc_auc: float,
        save_path: str,
        class_name: str = 'anomaly'
    ) -> None:
        """
        Plot and save ROC curves.

        Args:
            gt_labels: Ground truth labels for images
            img_scores: Image-level anomaly scores
            gt_masks: Ground truth masks
            score_maps: Pixel-level anomaly score maps
            image_roc_auc: Pre-computed image ROC-AUC
            pixel_roc_auc: Pre-computed pixel ROC-AUC
            save_path: Path to save figure
            class_name: Name of the class for labeling
        """
        if not self.save_roc:
            return

        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # Image-level ROC curve
        fpr, tpr, _ = roc_curve(gt_labels.astype(int), img_scores)
        ax[0].plot(fpr, tpr, label=f'Image ROC AUC: {image_roc_auc:.3f}')
        ax[0].set_title('Image-level ROC Curve')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].legend()

        # Pixel-level ROC curve
        fpr, tpr, _ = roc_curve(gt_masks.flatten().astype(int), score_maps.flatten())
        ax[1].plot(fpr, tpr, label=f'{class_name} Pixel ROC AUC: {pixel_roc_auc:.3f}')
        ax[1].set_title('Pixel-level ROC Curve')
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].legend()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curves to {save_path}")

    def plot_anomaly_heatmaps(
        self,
        test_imgs: List[np.ndarray],
        scores: np.ndarray,
        gt_masks: List[np.ndarray],
        threshold: float,
        save_dir: str,
        class_name: str = 'anomaly'
    ) -> None:
        """
        Generate comprehensive visualization of anomaly detection results.

        Creates a 5-panel figure for each test image showing:
        1. Original image
        2. Ground truth mask
        3. Predicted anomaly heatmap (overlaid on image)
        4. Binary predicted mask (thresholded)
        5. Segmentation result (boundaries marked on image)

        Args:
            test_imgs: List of test images (normalized)
            scores: Anomaly score maps of shape (N, H, W)
            gt_masks: List of ground truth masks
            threshold: Threshold for binarizing anomaly scores
            save_dir: Directory to save visualizations
            class_name: Name prefix for saved files
        """
        if not self.save_heatmaps:
            return

        num = len(scores)
        if self.max_samples is not None:
            num = min(num, self.max_samples)

        vmax = scores.max() * 255.
        vmin = scores.min() * 255.

        for i in range(num):
            # Denormalize image for visualization
            img = denormalization(test_imgs[i])

            # Prepare ground truth mask
            gt = gt_masks[i].transpose(1, 2, 0).squeeze()

            # Scale heatmap to 0-255
            heat_map = scores[i] * 255

            # Create binary mask by thresholding
            mask = scores[i].copy()
            mask[mask > threshold] = 1
            mask[mask <= threshold] = 0

            # Apply morphological opening to clean up mask
            kernel = morphology.disk(self.kernel_size)
            mask = morphology.opening(mask, kernel)
            mask *= 255

            # Create segmentation visualization with boundaries
            vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

            # Create 5-panel figure
            fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
            fig_img.subplots_adjust(right=0.9)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

            # Hide axes for all subplots
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)

            # Panel 1: Original image
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')

            # Panel 2: Ground truth
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')

            # Panel 3: Heatmap overlay
            ax_img[2].imshow(img, cmap='gray', interpolation='none')
            ax = ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
            ax_img[2].title.set_text('Predicted heat map')

            # Panel 4: Binary mask
            ax_img[3].imshow(mask, cmap='gray')
            ax_img[3].title.set_text('Predicted mask')

            # Panel 5: Segmentation boundaries
            ax_img[4].imshow(vis_img)
            ax_img[4].title.set_text('Segmentation result')

            # Add colorbar
            left, bottom, width, height = 0.92, 0.15, 0.015, 0.70
            cbar_ax = fig_img.add_axes([left, bottom, width, height])
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            cb.set_label('Anomaly Score', fontdict={
                'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8
            })

            # Save figure
            fig_img.savefig(
                os.path.join(save_dir, f'{class_name}_{i}.png'),
                dpi=100,
                bbox_inches='tight'
            )
            plt.close()

        print(f"Saved {num} heatmap visualizations to {save_dir}")

    def plot_results(
        self,
        scores: np.ndarray,
        test_dataset,
        results: Dict,
        save_dir: str,
        gt_labels: np.ndarray = None,
        img_scores: np.ndarray = None
    ) -> None:
        """
        Generate all visualizations.

        Args:
            scores: Anomaly score maps
            test_dataset: Test dataset
            results: Evaluation results
            save_dir: Directory to save visualizations
            gt_labels: Ground truth labels for ROC curves (optional)
            img_scores: Image-level scores for ROC curves (optional)
        """
        if not self.enabled:
            return

        # Extract test images and ground truth masks from dataset
        test_imgs = []
        gt_masks = []

        for i in range(len(test_dataset)):
            img, mask = test_dataset[i]
            test_imgs.append(img.numpy())
            gt_masks.append(mask.numpy())

        gt_masks_array = np.asarray(gt_masks)

        # Plot ROC curves if data provided
        if gt_labels is not None and img_scores is not None:
            self.plot_roc_curves(
                gt_labels,
                img_scores,
                gt_masks_array,
                scores,
                results['image_roc_auc'],
                results['pixel_roc_auc'],
                os.path.join(save_dir, 'roc_curves.png')
            )

        # Plot anomaly heatmaps
        self.plot_anomaly_heatmaps(
            test_imgs,
            scores,
            gt_masks,
            results.get('threshold', 0.5),
            save_dir
        )
