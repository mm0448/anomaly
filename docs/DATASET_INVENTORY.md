# Dataset Inventory

This document catalogs all available datasets for anomaly detection training and testing.

## Dataset Summary

| Dataset | Train Samples | Test Good | Test Anomalous | Anomaly Types |
|---------|--------------|-----------|----------------|---------------|
| bracket_black | 289 | 32 | 47 | hole, scratches |
| bracket_brown | 185 | 26 | 51 | bend_and_parts_mismatch, parts_mismatch |
| bracket_white | 110 | 30 | 30 | defective_painting, scratches |
| connector | 128 | 30 | 14 | parts_mismatch |
| metal_plate | 54 | 26 | 71 | major_rust, scratches, total_rust |
| tubes | 122 | 32 | 69 | anomalous |

## Detailed Dataset Information

### 1. bracket_black

**Description**: Black bracket components with various defects

**Training**:
- Normal samples: 289 (good)

**Testing**:
- Good samples: 32
- Anomalous samples: 47
  - `hole`: 12 samples - Holes or punctures in the bracket
  - `scratches`: 35 samples - Surface scratches and marks

**Paths**:
```
data/anomaly_dataset/bracket_black/
├── train/good/          (289 images)
├── test/
│   ├── good/            (32 images)
│   ├── hole/            (12 images)
│   └── scratches/       (35 images)
└── ground_truth/
    ├── hole/            (12 masks)
    └── scratches/       (35 masks)
```

### 2. bracket_brown

**Description**: Brown bracket components with assembly defects

**Training**:
- Normal samples: 185 (good)

**Testing**:
- Good samples: 26
- Anomalous samples: 51
  - `bend_and_parts_mismatch`: 17 samples - Combined bending and wrong parts
  - `parts_mismatch`: 34 samples - Incorrect or misassembled components

**Paths**:
```
data/anomaly_dataset/bracket_brown/
├── train/good/          (185 images)
├── test/
│   ├── good/            (26 images)
│   ├── bend_and_parts_mismatch/  (17 images)
│   └── parts_mismatch/  (34 images)
└── ground_truth/
    ├── bend_and_parts_mismatch/  (17 masks)
    └── parts_mismatch/  (34 masks)
```

### 3. bracket_white

**Description**: White bracket components with surface defects

**Training**:
- Normal samples: 110 (good)

**Testing**:
- Good samples: 30
- Anomalous samples: 30
  - `defective_painting`: 13 samples - Paint defects and coating issues
  - `scratches`: 17 samples - Surface scratches

**Paths**:
```
data/anomaly_dataset/bracket_white/
├── train/good/          (110 images)
├── test/
│   ├── good/            (30 images)
│   ├── defective_painting/  (13 images)
│   └── scratches/       (17 images)
└── ground_truth/
    ├── defective_painting/  (13 masks)
    └── scratches/       (17 masks)
```

### 4. connector

**Description**: Connector components with assembly defects

**Training**:
- Normal samples: 128 (good)

**Testing**:
- Good samples: 30
- Anomalous samples: 14
  - `parts_mismatch`: 14 samples - Wrong or misassembled parts

**Paths**:
```
data/anomaly_dataset/connector/
├── train/good/          (128 images)
├── test/
│   ├── good/            (30 images)
│   └── parts_mismatch/  (14 images)
└── ground_truth/
    └── parts_mismatch/  (14 masks)
```

### 5. metal_plate

**Description**: Metal plates with corrosion and surface defects

**Training**:
- Normal samples: 54 (good)

**Testing**:
- Good samples: 26
- Anomalous samples: 71
  - `major_rust`: 14 samples - Significant rust and corrosion
  - `scratches`: 34 samples - Surface scratches
  - `total_rust`: 23 samples - Complete rust coverage

**Paths**:
```
data/anomaly_dataset/metal_plate/
├── train/good/          (54 images)
├── test/
│   ├── good/            (26 images)
│   ├── major_rust/      (14 images)
│   ├── scratches/       (34 images)
│   └── total_rust/      (23 images)
└── ground_truth/
    ├── major_rust/      (14 masks)
    ├── scratches/       (34 masks)
    └── total_rust/      (23 masks)
```

### 6. tubes

**Description**: Tube components with general anomalies

**Training**:
- Normal samples: 122 (good)

**Testing**:
- Good samples: 32
- Anomalous samples: 69
  - `anomalous`: 69 samples - General defects (not categorized)

**Paths**:
```
data/anomaly_dataset/tubes/
├── train/good/          (122 images)
├── test/
│   ├── good/            (32 images)
│   └── anomalous/       (69 images)
└── ground_truth/
    └── anomalous/       (69 masks)
```

## Dataset Characteristics

### Sample Distribution

**Training samples per dataset**:
- Largest: bracket_black (289)
- Smallest: metal_plate (54)
- Average: 148

**Test anomalous samples per dataset**:
- Largest: metal_plate (71)
- Smallest: connector (14)
- Average: 47

### Anomaly Types

**Surface defects**:
- Scratches (3 datasets: bracket_black, bracket_white, metal_plate)
- Holes (1 dataset: bracket_black)
- Defective painting (1 dataset: bracket_white)

**Corrosion**:
- Major rust (1 dataset: metal_plate)
- Total rust (1 dataset: metal_plate)

**Assembly defects**:
- Parts mismatch (3 datasets: bracket_brown, bracket_white, connector)
- Bend and parts mismatch (1 dataset: bracket_brown)

**General**:
- Anomalous (1 dataset: tubes - mixed defects)

## Usage Notes

### Multiple Anomaly Types

For datasets with multiple anomaly types (bracket_black, bracket_brown, bracket_white, metal_plate), you can:

1. **Test all anomalies together**: Combine all anomalous test sets
2. **Test per anomaly type**: Evaluate each defect category separately

Configuration files are provided for both approaches.

### Ground Truth Masks

All anomalous test samples have corresponding pixel-level ground truth masks for:
- Pixel-level evaluation (ROC-AUC)
- Anomaly localization visualization
- Threshold optimization

### Recommended Configurations

**For robust evaluation**:
- Use dataset-specific configs that test all anomaly types together
- Example: `configs/datasets/metal_plate_all.yaml`

**For per-defect analysis**:
- Use anomaly-specific configs
- Example: `configs/datasets/metal_plate_scratches.yaml`

**For benchmarking**:
- Test on all datasets with same hyperparameters
- Compare performance across different object classes

## Generating Configuration Files

To generate config files for all datasets and anomaly types:

```bash
python scripts/generate_dataset_configs.py
```

This creates:
- Per-dataset configs (all anomalies): `configs/datasets/{dataset}_all.yaml`
- Per-anomaly configs: `configs/datasets/{dataset}_{anomaly}.yaml`

## Dataset Statistics

### Total Counts

- **Total training images**: 888
- **Total test good images**: 176
- **Total test anomalous images**: 282
- **Total test images**: 458

### Class Balance

Most datasets have imbalanced test sets (more good samples than anomalies per type), which is realistic for industrial scenarios where defects are rare.

### Image Format

- All images: PNG or JPG
- Ground truth masks: PNG (binary masks with `_mask.png` suffix)

## Future Datasets

To add a new dataset:

1. Create directory structure:
   ```
   data/anomaly_dataset/new_dataset/
   ├── train/good/
   ├── test/{good,anomaly_type}/
   └── ground_truth/{anomaly_type}/
   ```

2. Update this inventory document

3. Generate config file:
   ```bash
   python scripts/generate_dataset_configs.py
   ```

4. Test the new dataset:
   ```bash
   python scripts/train_evaluate.py --config configs/datasets/new_dataset_all.yaml
   ```
