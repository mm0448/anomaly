#!/usr/bin/env python
"""
Modular entry point for anomaly detection.

Usage:
    python scripts/train_evaluate.py --config configs/default_padim.yaml
    python scripts/train_evaluate.py --config configs/padim_resnet50.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from anomaly_detector.pipeline import AnomalyDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Run anomaly detection training and evaluation',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = AnomalyDetectionPipeline(str(config_path))
        results = pipeline.run()

        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        if 'image_roc_auc' in results:
            print(f"Image-level ROC-AUC: {results['image_roc_auc']:.3f}")
        if 'pixel_roc_auc' in results:
            print(f"Pixel-level ROC-AUC: {results['pixel_roc_auc']:.3f}")
        if 'threshold' in results:
            print(f"Optimal threshold: {results['threshold']:.3f}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
