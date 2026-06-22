"""Evaluator for anomaly detection results."""

import numpy as np
from typing import Dict
from .metrics import (
    compute_image_roc_auc,
    compute_pixel_roc_auc,
    find_optimal_threshold
)


class Evaluator:
    """
    Evaluator for anomaly detection results.

    Computes various metrics including:
    - Image-level ROC-AUC
    - Pixel-level ROC-AUC
    - Optimal threshold
    """

    def __init__(self, config: Dict):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration dictionary with keys:
                - metrics: List of metrics to compute
                - threshold_method: Method for finding threshold ('f1', 'precision', 'recall')
        """
        self.config = config
        self.metrics = config.get('metrics', ['roc_auc', 'pixel_roc_auc', 'f1'])
        self.threshold_method = config.get('threshold_method', 'f1')

    def evaluate(
        self,
        scores: np.ndarray,
        test_dataset,
        test_loader
    ) -> tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate anomaly detection results.

        Args:
            scores: Anomaly score maps of shape (N, H, W)
            test_dataset: Test dataset with n_anomalous attribute
            test_loader: Test dataloader for extracting ground truth

        Returns:
            Dictionary of evaluation results
        """
        # Extract ground truth labels and masks
        gt_list = []
        gt_mask_list = []

        n_anomalous = test_dataset.n_anomalous

        for idx, (_, mask) in enumerate(test_loader):
            mask_np = mask.cpu().detach().numpy()
            mask_binary = (mask_np > 0.5).astype(np.float32)
            gt_mask_list.extend(mask_binary)

            # Assign labels based on dataset source
            for i in range(mask.shape[0]):
                current_idx = idx * test_loader.batch_size + i
                if current_idx < n_anomalous:
                    gt_list.append(1)  # Anomalous
                else:
                    gt_list.append(0)  # Normal

        gt_list = np.asarray(gt_list)
        gt_mask = np.asarray(gt_mask_list)

        results = {}

        # Image-level evaluation
        img_scores = None
        if 'roc_auc' in self.metrics:
            img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
            img_roc_auc, _, _ = compute_image_roc_auc(gt_list, img_scores)
            results['image_roc_auc'] = float(img_roc_auc)

        # Pixel-level evaluation
        if 'pixel_roc_auc' in self.metrics:
            pixel_roc_auc, _, _ = compute_pixel_roc_auc(gt_mask, scores)
            results['pixel_roc_auc'] = float(pixel_roc_auc)

        # Find optimal threshold
        if 'f1' in self.metrics:
            threshold = find_optimal_threshold(
                gt_mask,
                scores,
                method=self.threshold_method
            )
            results['threshold'] = float(threshold)

        # Return results plus data needed for visualizations
        return results, gt_list, img_scores, gt_mask
