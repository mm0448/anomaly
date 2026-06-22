#!/usr/bin/env python
"""
Generate configuration files for all datasets and anomaly types.

This script scans the data directory and creates YAML configuration files for:
1. Each dataset with all anomaly types combined
2. Each individual anomaly type per dataset

Usage:
    python scripts/generate_dataset_configs.py
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def scan_datasets(data_dir: Path) -> Dict[str, Dict]:
    """
    Scan datasets directory and extract structure.

    Returns:
        Dictionary mapping dataset names to their structure
    """
    datasets = {}

    for dataset_path in sorted(data_dir.iterdir()):
        if not dataset_path.is_dir() or dataset_path.name.startswith('.'):
            continue

        dataset_name = dataset_path.name

        # Get training directory
        train_dir = dataset_path / "train" / "good"
        if not train_dir.exists():
            print(f"Warning: No training data found for {dataset_name}")
            continue

        # Get test anomaly types
        test_dir = dataset_path / "test"
        ground_truth_dir = dataset_path / "ground_truth"

        anomaly_types = []
        if test_dir.exists():
            for test_subdir in sorted(test_dir.iterdir()):
                if test_subdir.is_dir() and test_subdir.name != "good":
                    anomaly_types.append(test_subdir.name)

        # Get good test directory
        test_good_dir = test_dir / "good" if test_dir.exists() else None

        datasets[dataset_name] = {
            'train_dir': str(train_dir),
            'test_good_dir': str(test_good_dir) if test_good_dir else None,
            'anomaly_types': anomaly_types,
            'test_dir': str(test_dir),
            'ground_truth_dir': str(ground_truth_dir)
        }

    return datasets


def create_config_all_anomalies(
    dataset_name: str,
    dataset_info: Dict,
    output_dir: Path
) -> None:
    """
    Create config file for dataset with all anomaly types combined.

    For datasets with multiple anomaly types, this creates a config that
    evaluates all anomalies together (common use case).
    """
    if not dataset_info['anomaly_types']:
        print(f"Skipping {dataset_name}: no anomaly types found")
        return

    # Use first anomaly type as primary for paths
    first_anomaly = dataset_info['anomaly_types'][0]

    config = {
        'experiment': {
            'name': f"{dataset_name}_padim_all_anomalies",
            'seed': 42,
            'device': 'cpu'
        },
        'data': {
            'train_dir': dataset_info['train_dir'],
            'test_dir_bad': f"{dataset_info['test_dir']}/{first_anomaly}",
            'test_dir_bad_masks': f"{dataset_info['ground_truth_dir']}/{first_anomaly}",
            'test_dir_good': dataset_info['test_good_dir'],
            'class_name': dataset_name,
            'transforms': {
                'resize': 256,
                'center_crop': 224,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'batch_size': 8,
            'num_workers': 0,
            'shuffle': False
        },
        'encoder': {
            'type': 'resnet',
            'backbone': 'resnet18',
            'layers': ['layer1', 'layer2', 'layer3'],
            'pretrained': True
        },
        'algorithm': {
            'type': 'padim',
            'random_seed': 42,
            'padim': {
                'dimension_reduction': 100,
                'total_dimensions': 448,
                'gaussian_smoothing_sigma': 4,
                'covariance_regularization': 0.01
            }
        },
        'cache': {
            'enabled': True,
            'cache_dir': f'outputs/cache/{dataset_name}',
            'train_features_file': 'train_features.pkl',
            'use_cached_if_exists': True
        },
        'evaluation': {
            'metrics': ['roc_auc', 'pixel_roc_auc', 'f1'],
            'threshold_method': 'f1'
        },
        'visualization': {
            'enabled': True,
            'save_roc_curves': True,
            'save_heatmaps': True,
            'max_samples': None,
            'morphology_kernel_size': 4
        },
        'output': {
            'save_dir': f'outputs/{dataset_name}_all_anomalies',
            'model_dir': f'outputs/{dataset_name}_all_anomalies/models',
            'plots_dir': f'outputs/{dataset_name}_all_anomalies/visualizations',
            'results_file': f'outputs/{dataset_name}_all_anomalies/results.json'
        }
    }

    # Add comment about multiple anomaly types
    if len(dataset_info['anomaly_types']) > 1:
        config['_comment'] = (
            f"This config tests all anomaly types together. "
            f"Available types: {', '.join(dataset_info['anomaly_types'])}. "
            f"To test individually, use the per-anomaly config files."
        )

    output_file = output_dir / f"{dataset_name}_all.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {output_file}")


def create_config_per_anomaly(
    dataset_name: str,
    dataset_info: Dict,
    anomaly_type: str,
    output_dir: Path
) -> None:
    """
    Create config file for a specific anomaly type.
    """
    config = {
        'experiment': {
            'name': f"{dataset_name}_{anomaly_type}_padim",
            'seed': 42,
            'device': 'cpu'
        },
        'data': {
            'train_dir': dataset_info['train_dir'],
            'test_dir_bad': f"{dataset_info['test_dir']}/{anomaly_type}",
            'test_dir_bad_masks': f"{dataset_info['ground_truth_dir']}/{anomaly_type}",
            'test_dir_good': dataset_info['test_good_dir'],
            'class_name': f"{dataset_name}_{anomaly_type}",
            'transforms': {
                'resize': 256,
                'center_crop': 224,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            'batch_size': 8,
            'num_workers': 0,
            'shuffle': False
        },
        'encoder': {
            'type': 'resnet',
            'backbone': 'resnet18',
            'layers': ['layer1', 'layer2', 'layer3'],
            'pretrained': True
        },
        'algorithm': {
            'type': 'padim',
            'random_seed': 42,
            'padim': {
                'dimension_reduction': 100,
                'total_dimensions': 448,
                'gaussian_smoothing_sigma': 4,
                'covariance_regularization': 0.01
            }
        },
        'cache': {
            'enabled': True,
            'cache_dir': f'outputs/cache/{dataset_name}',
            'train_features_file': 'train_features.pkl',
            'use_cached_if_exists': True
        },
        'evaluation': {
            'metrics': ['roc_auc', 'pixel_roc_auc', 'f1'],
            'threshold_method': 'f1'
        },
        'visualization': {
            'enabled': True,
            'save_roc_curves': True,
            'save_heatmaps': True,
            'max_samples': None,
            'morphology_kernel_size': 4
        },
        'output': {
            'save_dir': f'outputs/{dataset_name}_{anomaly_type}',
            'model_dir': f'outputs/{dataset_name}_{anomaly_type}/models',
            'plots_dir': f'outputs/{dataset_name}_{anomaly_type}/visualizations',
            'results_file': f'outputs/{dataset_name}_{anomaly_type}/results.json'
        }
    }

    safe_name = anomaly_type.replace('/', '_').replace(' ', '_')
    output_file = output_dir / f"{dataset_name}_{safe_name}.yaml"

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created: {output_file}")


def main():
    """Main function to generate all config files."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "anomaly_dataset"
    output_dir = project_root / "configs" / "datasets"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Dataset Configuration Generator")
    print("="*60)
    print()

    # Scan datasets
    print(f"Scanning datasets in: {data_dir}")
    datasets = scan_datasets(data_dir)
    print(f"Found {len(datasets)} datasets")
    print()

    # Generate configs
    total_configs = 0

    for dataset_name, dataset_info in datasets.items():
        print(f"\nProcessing: {dataset_name}")
        print(f"  Anomaly types: {', '.join(dataset_info['anomaly_types'])}")

        # Create "all anomalies" config
        create_config_all_anomalies(dataset_name, dataset_info, output_dir)
        total_configs += 1

        # Create per-anomaly configs
        for anomaly_type in dataset_info['anomaly_types']:
            create_config_per_anomaly(
                dataset_name,
                dataset_info,
                anomaly_type,
                output_dir
            )
            total_configs += 1

    print()
    print("="*60)
    print(f"Generation complete!")
    print(f"Created {total_configs} configuration files in {output_dir}")
    print("="*60)
    print()
    print("Usage examples:")
    print("  # Test all anomalies together")
    print(f"  python scripts/train_evaluate.py --config configs/datasets/metal_plate_all.yaml")
    print()
    print("  # Test specific anomaly type")
    print(f"  python scripts/train_evaluate.py --config configs/datasets/metal_plate_scratches.yaml")
    print()


if __name__ == "__main__":
    main()
