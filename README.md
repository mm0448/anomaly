# Anomaly Detection Framework

A modular, extensible framework for industrial anomaly detection using deep learning.

<img width="3569" height="2969" alt="image" src="https://github.com/user-attachments/assets/9760f01c-30ca-40ce-9395-fb7b261fb9a0" />

<img width="2369" height="1768" alt="image" src="https://github.com/user-attachments/assets/70a8a614-1b72-4082-b38d-213be3e32808" />

### Installation

```bash
# Install package
pip install -e .
```

### Usage

```bash
# Run with default configuration
python scripts/train_evaluate.py --config configs/default_padim.yaml
```

### Directory Structure

```
.
├── anomaly_detector/          # Main package
│   ├── core/                  # Abstract base classes
│   ├── encoders/              # Feature extractors (ResNet, etc.)
│   ├── algorithms/            # Anomaly detection methods (PaDiM, etc.)
│   ├── data/                  # Data loading and preprocessing
│   ├── evaluation/            # Metrics and evaluation
│   ├── visualization/         # Plotting utilities
│   ├── utils/                 # Configuration, caching, seeds
│   └── pipeline.py            # Main orchestration
├── configs/                   # YAML configurations
│   ├── default_padim.yaml
│   ├── encoders/
│   └── algorithms/
├── scripts/
│   ├── run_padim.py          # Original script (preserved)
│   └── train_evaluate.py     # New modular entry point
├── data/                      # Your datasets
├── outputs/                   # Results, models, visualizations
└── docs/                      # Documentation
```

## Usage Examples

### 1. Default Configuration

```bash
python scripts/train_evaluate.py --config configs/default_padim.yaml
```

Uses ResNet18 encoder with PaDiM algorithm (dimension reduction to 100).

### 1b. Test All Available Datasets

Run on any of the 6 available datasets (see [Dataset Inventory](docs/DATASET_INVENTORY.md)):

```bash
# Test all datasets
python scripts/train_evaluate.py --config configs/datasets/bracket_black_all.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_brown_all.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_white_all.yaml
python scripts/train_evaluate.py --config configs/datasets/connector_all.yaml
python scripts/train_evaluate.py --config configs/datasets/metal_plate_all.yaml
python scripts/train_evaluate.py --config configs/datasets/tubes_all.yaml

# Or test specific anomaly types
python scripts/train_evaluate.py --config configs/datasets/metal_plate_scratches.yaml
python scripts/train_evaluate.py --config configs/datasets/bracket_black_hole.yaml
```

See [Dataset Configs Reference](docs/DATASET_CONFIGS.md) for all 17 available configurations.

### 2. Different Encoder

```bash
python scripts/train_evaluate.py --config configs/encoders/resnet50.yaml
```

Uses ResNet50 (larger backbone, better performance).

### 3. No Dimension Reduction

```bash
python scripts/train_evaluate.py --config configs/algorithms/padim_full.yaml
```

Uses all 448 dimensions

### 4. Pipeline Usage

```python
from anomaly_detector.pipeline import AnomalyDetectionPipeline

# Initialize and run pipeline
pipeline = AnomalyDetectionPipeline('configs/default_padim.yaml')
results = pipeline.run()

# Access results
print(f"Image ROC-AUC: {results['image_roc_auc']:.3f}")
print(f"Pixel ROC-AUC: {results['pixel_roc_auc']:.3f}")
print(f"Threshold: {results['threshold']:.3f}")
```

### 5. Custom Configuration

Create a new YAML file:

```yaml
# my_config.yaml
experiment:
  name: "my_experiment"
  seed: 42
  device: "cpu"

data:
  train_dir: "data/my_dataset/train/good"
  test_dir_bad: "data/my_dataset/test/defect"
  test_dir_bad_masks: "data/my_dataset/ground_truth/defect"
  test_dir_good: "data/my_dataset/test/good"
  class_name: "my_class"
  batch_size: 16

encoder:
  type: "resnet"
  backbone: "resnet34"
  layers: ["layer1", "layer2", "layer3"]

algorithm:
  type: "padim"
  padim:
    dimension_reduction: 150

output:
  save_dir: "outputs/my_experiment"
```

Run it:
```bash
python scripts/train_evaluate.py --config my_config.yaml
```

## Configuration Reference

### Experiment Settings

```yaml
experiment:
  name: "experiment_name"     # Identifier for this run
  seed: 42                    # Random seed for reproducibility
  device: "cpu"               # "cpu" or "cuda:0"
```

### Data Settings

```yaml
data:
  train_dir: "path/to/train/good"
  test_dir_bad: "path/to/test/anomalous"
  test_dir_bad_masks: "path/to/ground_truth"
  test_dir_good: "path/to/test/good"
  class_name: "object_class"
  
  transforms:
    resize: 256
    center_crop: 224
    normalize:
      mean: [0.485, 0.456, 0.406]  # ImageNet mean
      std: [0.229, 0.224, 0.225]   # ImageNet std
  
  batch_size: 8
  num_workers: 0
  shuffle: false
```

### Encoder Settings

```yaml
encoder:
  type: "resnet"                           # Encoder type
  backbone: "resnet18"                     # resnet18, resnet34, resnet50, resnet101
  layers: ["layer1", "layer2", "layer3"]  # Layers to extract features from
  pretrained: true                         # Use ImageNet pretrained weights
```

### Algorithm Settings

```yaml
algorithm:
  type: "padim"
  random_seed: 42
  
  padim:
    dimension_reduction: 100              # null for no reduction
    total_dimensions: 448                 # Auto-computed from encoder
    gaussian_smoothing_sigma: 4           # Spatial smoothing
    covariance_regularization: 0.01       # Numerical stability
```

### Cache Settings

```yaml
cache:
  enabled: true
  cache_dir: "outputs/cache"
  train_features_file: "train_features.pkl"
  use_cached_if_exists: true
```

To force re-extraction: `rm -rf outputs/cache`

### Evaluation Settings

```yaml
evaluation:
  metrics: ["roc_auc", "pixel_roc_auc", "f1"]
  threshold_method: "f1"  # f1, precision, or recall
```

### Visualization Settings

```yaml
visualization:
  enabled: true
  save_roc_curves: true
  save_heatmaps: true
  max_samples: null              # null = all, or specify number
  morphology_kernel_size: 4      # For mask cleanup
```

### Output Settings

```yaml
output:
  save_dir: "outputs/experiment"
  model_dir: "outputs/experiment/models"
  plots_dir: "outputs/experiment/visualizations"
  results_file: "outputs/experiment/results.json"
```

## Output File Structure

```
outputs/experiment/
├── config.yaml                    # Saved configuration
├── results.json                   # Evaluation metrics
├── models/
│   └── model.pkl                  # Fitted model parameters
└── visualizations/
    ├── roc_curves.png             # ROC curves
    ├── anomaly_0.png              # Per-sample visualizations
    ├── anomaly_1.png
    └── ...
```

## Extending the Framework

### Adding a New Encoder

1. **Create encoder class**:

```python
# anomaly_detector/encoders/my_encoder.py
from ..core.base_encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    def build_model(self):
        # Load your model
        pass
    
    def extract_features(self, dataloader):
        # Extract features
        pass
    
    def get_feature_dimensions(self):
        # Return dimensions
        return {'layer1': 128, 'layer2': 256}
```

2. **Register in factory**:

```python
# anomaly_detector/encoders/encoder_factory.py
def create_encoder(config, device='cpu'):
    if config['type'] == 'my_encoder':
        return MyEncoder(...)
```

3. **Create config**:

```yaml
# configs/encoders/my_encoder.yaml
encoder:
  type: "my_encoder"
  # Add your parameters
```

### Adding a New Algorithm

1. **Create algorithm class**:

```python
# anomaly_detector/algorithms/my_algorithm.py
from ..core.base_algorithm import BaseAnomalyAlgorithm

class MyAlgorithm(BaseAnomalyAlgorithm):
    def fit(self, features):
        # Learn from training features
        pass
    
    def predict(self, features):
        # Return anomaly scores
        pass
    
    def save(self, path):
        # Save model
        pass
    
    def load(self, path):
        # Load model
        pass
```

2. **Register in factory**:

```python
# anomaly_detector/algorithms/algorithm_factory.py
def create_algorithm(config, encoder_dims=None):
    if config['type'] == 'my_algorithm':
        return MyAlgorithm(...)
```

3. **Create config**:

```yaml
# configs/algorithms/my_algorithm.yaml
algorithm:
  type: "my_algorithm"
  # Add your parameters
```
