# Dataset Configuration Files

This document provides a reference for all available dataset configuration files.

## Quick Reference

### All Datasets - Combined Anomalies

Test all anomaly types together (recommended for overall performance evaluation):

```bash
# Bracket datasets
python scripts/train_evaluate.py --config configs/datasets/bracket_black_all.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_brown_all.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_white_all.yaml

# Connector
python scripts/train_evaluate.py --config configs/datasets/connector_all.yaml

# Metal plate
python scripts/train_evaluate.py --config configs/datasets/metal_plate_all.yaml

# Tubes
python scripts/train_evaluate.py --config configs/datasets/tubes_all.yaml
```

### Per-Anomaly Type Configs

For detailed analysis of specific defect types:

**bracket_black**:
```bash
python scripts/train_evaluate.py --config configs/datasets/bracket_black_hole.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_black_scratches.yaml
```

**bracket_brown**:
```bash
python scripts/train_evaluate.py --config configs/datasets/bracket_brown_bend_and_parts_mismatch.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_brown_parts_mismatch.yaml
```

**bracket_white**:
```bash
python scripts/train_evaluate.py --config configs/datasets/bracket_white_defective_painting.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_white_scratches.yaml
```

**connector**:
```bash
python scripts/train_evaluate.py --config configs/datasets/connector_parts_mismatch.yaml
```

**metal_plate**:
```bash
python scripts/train_evaluate.py --config configs/datasets/metal_plate_major_rust.yaml
python scripts/train_evaluate.py --config configs/datasets/metal_plate_scratches.yaml
python scripts/train_evaluate.py --config configs/datasets/metal_plate_total_rust.yaml
```

**tubes**:
```bash
python scripts/train_evaluate.py --config configs/datasets/tubes_anomalous.yaml
```

## Configuration Matrix

| Dataset | Config File | Anomaly Types | Train Samples | Test Anomalous |
|---------|-------------|---------------|---------------|----------------|
| **bracket_black** | | | | |
| | bracket_black_all.yaml | hole, scratches | 289 | 47 |
| | bracket_black_hole.yaml | hole | 289 | 12 |
| | bracket_black_scratches.yaml | scratches | 289 | 35 |
| **bracket_brown** | | | | |
| | bracket_brown_all.yaml | bend_and_parts_mismatch, parts_mismatch | 185 | 51 |
| | bracket_brown_bend_and_parts_mismatch.yaml | bend_and_parts_mismatch | 185 | 17 |
| | bracket_brown_parts_mismatch.yaml | parts_mismatch | 185 | 34 |
| **bracket_white** | | | | |
| | bracket_white_all.yaml | defective_painting, scratches | 110 | 30 |
| | bracket_white_defective_painting.yaml | defective_painting | 110 | 13 |
| | bracket_white_scratches.yaml | scratches | 110 | 17 |
| **connector** | | | | |
| | connector_all.yaml | parts_mismatch | 128 | 14 |
| | connector_parts_mismatch.yaml | parts_mismatch | 128 | 14 |
| **metal_plate** | | | | |
| | metal_plate_all.yaml | major_rust, scratches, total_rust | 54 | 71 |
| | metal_plate_major_rust.yaml | major_rust | 54 | 14 |
| | metal_plate_scratches.yaml | scratches | 54 | 34 |
| | metal_plate_total_rust.yaml | total_rust | 54 | 23 |
| **tubes** | | | | |
| | tubes_all.yaml | anomalous | 122 | 69 |
| | tubes_anomalous.yaml | anomalous | 122 | 69 |

**Total**: 17 configuration files (6 "all" configs + 11 per-anomaly configs)

## Configuration Details

All configuration files include:

### Common Parameters

```yaml
experiment:
  name: "{dataset}_{anomaly}_padim"
  seed: 42
  device: "cpu"

encoder:
  type: "resnet"
  backbone: "resnet18"
  layers: ["layer1", "layer2", "layer3"]
  pretrained: true

algorithm:
  type: "padim"
  random_seed: 42
  padim:
    dimension_reduction: 100
    total_dimensions: 448
    gaussian_smoothing_sigma: 4
    covariance_regularization: 0.01

cache:
  enabled: true
  cache_dir: "outputs/cache/{dataset}"
  train_features_file: "train_features.pkl"
  use_cached_if_exists: true
```

### Dataset-Specific Paths

Each config automatically configures:
- `data.train_dir`: Points to training good samples
- `data.test_dir_bad`: Points to anomalous test samples
- `data.test_dir_bad_masks`: Points to ground truth masks
- `data.test_dir_good`: Points to good test samples
- `output.*`: Results saved to `outputs/{dataset}_{anomaly}/`

## Use Cases

### 1. Benchmark All Datasets

Test PaDiM on all datasets to compare performance:

```bash
# Run all datasets
for config in configs/datasets/*_all.yaml; do
    echo "Testing: $config"
    python scripts/train_evaluate.py --config "$config"
done
```

Results will be saved to separate directories:
- `outputs/bracket_black_all_anomalies/`
- `outputs/bracket_brown_all_anomalies/`
- etc.

### 2. Per-Defect Analysis

Analyze how well the model detects specific defect types:

```bash
# Test all metal plate defect types separately
python scripts/train_evaluate.py --config configs/datasets/metal_plate_major_rust.yaml
python scripts/train_evaluate.py --config configs/datasets/metal_plate_scratches.yaml
python scripts/train_evaluate.py --config configs/datasets/metal_plate_total_rust.yaml
```

Compare results to identify which defects are easier/harder to detect.

### 3. Feature Sharing

Since cache is per-dataset, training features are shared:

```bash
# First run extracts and caches features
python scripts/train_evaluate.py --config configs/datasets/metal_plate_scratches.yaml

# Second run reuses cached features (much faster)
python scripts/train_evaluate.py --config configs/datasets/metal_plate_major_rust.yaml
```

Both use the same training data → same cached features.

### 4. Hyperparameter Tuning

Modify a config and test:

```bash
# Copy and modify
cp configs/datasets/bracket_black_all.yaml configs/my_experiment.yaml

# Edit: change dimension_reduction from 100 to 200
# Edit: change gaussian_smoothing_sigma from 4 to 2

# Test modified config
python scripts/train_evaluate.py --config configs/my_experiment.yaml
```

### 5. Different Encoders

Try ResNet50 on a specific dataset:

```bash
# Copy config
cp configs/datasets/tubes_all.yaml configs/tubes_resnet50.yaml

# Edit:
#   encoder.backbone: "resnet50"
#   algorithm.padim.dimension_reduction: 200
#   algorithm.padim.total_dimensions: 1792

python scripts/train_evaluate.py --config configs/tubes_resnet50.yaml
```

## Regenerating Configs

If you add new datasets or anomaly types:

```bash
python scripts/generate_dataset_configs.py
```

This will:
1. Scan `data/anomaly_dataset/` for all datasets
2. Detect training and test directories
3. Generate configs for all combinations
4. Preserve existing configs (overwrites with updates)

## Config Naming Convention

**Pattern**: `{dataset}_{anomaly}.yaml`

- `{dataset}`: Dataset name (e.g., `metal_plate`)
- `{anomaly}`: Either `all` (all types) or specific anomaly name

**Examples**:
- `metal_plate_all.yaml` - All anomalies together
- `metal_plate_scratches.yaml` - Only scratches
- `bracket_black_hole.yaml` - Only hole defects

## Output Organization

Each config saves results to a unique directory:

```
outputs/
├── bracket_black_all_anomalies/
│   ├── config.yaml
│   ├── results.json
│   ├── models/
│   └── visualizations/
├── bracket_black_hole/
│   ├── config.yaml
│   ├── results.json
│   ├── models/
│   └── visualizations/
└── ...
```

This organization allows:
- Running experiments in parallel
- Comparing results across configs
- No output conflicts

## Tips

### Performance Optimization

**Use GPU**:
```yaml
experiment:
  device: "cuda:0"  # Change from "cpu"
```

**Increase batch size** (if memory allows):
```yaml
data:
  batch_size: 16  # Change from 8
```

### Reducing Storage

**Disable visualizations** for large test sets:
```yaml
visualization:
  enabled: false  # or set max_samples: 10
```

**Disable caching** if disk space is limited:
```yaml
cache:
  enabled: false
```

### Quick Testing

**Test with fewer samples**:
```yaml
visualization:
  max_samples: 5  # Only visualize first 5
```

## Common Tasks

### Run on All Datasets Sequentially

```bash
#!/bin/bash
for config in configs/datasets/*_all.yaml; do
    echo "Running: $config"
    python scripts/train_evaluate.py --config "$config"
    echo "---"
done
```

### Compare Specific Anomaly Across Datasets

```bash
# Compare scratches detection across all datasets that have scratches
python scripts/train_evaluate.py --config configs/datasets/bracket_black_scratches.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_white_scratches.yaml
python scripts/train_evaluate.py --config configs/datasets/metal_plate_scratches.yaml
```

### Collect Results

```bash
# Extract ROC-AUC scores from all results
for results in outputs/*/results.json; do
    dataset=$(dirname $results | xargs basename)
    roc_auc=$(jq -r '.image_roc_auc' $results)
    echo "$dataset: $roc_auc"
done
```

## See Also

- [Dataset Inventory](DATASET_INVENTORY.md) - Detailed dataset information
- [Configuration Reference](../README.md#configuration-reference) - Full config schema
- [Architecture](ARCHITECTURE.md) - System design details
