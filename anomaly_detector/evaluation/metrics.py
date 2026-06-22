"""Evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from typing import Tuple


def compute_image_roc_auc(
    gt_labels: np.ndarray,
    img_scores: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute image-level ROC-AUC score.

    Args:
        gt_labels: Ground truth binary labels (0=normal, 1=anomalous)
        img_scores: Anomaly scores per image

    Returns:
        Tuple of (roc_auc, fpr, tpr)
    """
    fpr, tpr, _ = roc_curve(gt_labels, img_scores)
    roc_auc = roc_auc_score(gt_labels, img_scores)
    return roc_auc, fpr, tpr


def compute_pixel_roc_auc(
    gt_masks: np.ndarray,
    score_maps: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute pixel-level ROC-AUC score.

    Args:
        gt_masks: Ground truth binary masks
        score_maps: Anomaly score maps

    Returns:
        Tuple of (roc_auc, fpr, tpr)
    """
    fpr, tpr, _ = roc_curve(gt_masks.flatten(), score_maps.flatten())
    roc_auc = roc_auc_score(gt_masks.flatten(), score_maps.flatten())
    return roc_auc, fpr, tpr


def find_optimal_threshold(
    gt_masks: np.ndarray,
    score_maps: np.ndarray,
    method: str = 'f1'
) -> float:
    """
    Find optimal threshold for binarizing anomaly scores.

    Args:
        gt_masks: Ground truth binary masks
        score_maps: Anomaly score maps
        method: Method for finding threshold ('f1', 'precision', 'recall')

    Returns:
        Optimal threshold value
    """
    precision, recall, thresholds = precision_recall_curve(
        gt_masks.flatten(),
        score_maps.flatten()
    )

    if method == 'f1':
        # Find threshold that maximizes F1 score
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0
        )
        optimal_idx = np.argmax(f1)
    elif method == 'precision':
        # Find threshold that maximizes precision
        optimal_idx = np.argmax(precision)
    elif method == 'recall':
        # Find threshold that maximizes recall
        optimal_idx = np.argmax(recall)
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
