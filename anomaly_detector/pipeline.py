"""Main pipeline for anomaly detection training and evaluation."""

import json
from pathlib import Path
from typing import Dict

from .utils.config import load_config, save_config
from .utils.seed import set_random_seeds
from .utils.caching import FeatureCache
from .data.data_loader import create_dataloaders
from .encoders.encoder_factory import create_encoder
from .algorithms.algorithm_factory import create_algorithm
from .evaluation.evaluator import Evaluator
from .visualization.anomaly_viz import AnomalyVisualizer


class AnomalyDetectionPipeline:
    """
    Main pipeline for anomaly detection training and evaluation.

    Orchestrates the full workflow:
    1. Load and prepare data
    2. Extract features using encoder
    3. Fit anomaly detection algorithm
    4. Evaluate performance
    5. Generate visualizations
    """

    def __init__(self, config_path: str):
        """
        Initialize pipeline.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = load_config(config_path)
        self._setup_environment()
        self._create_output_dirs()

    def _setup_environment(self):
        """Setup random seeds and device."""
        set_random_seeds(self.config['experiment']['seed'])
        self.device = self.config['experiment']['device']

    def _create_output_dirs(self):
        """Create output directories."""
        for key in ['save_dir', 'model_dir', 'plots_dir']:
            Path(self.config['output'][key]).mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        """
        Execute full pipeline: train, evaluate, visualize.

        Returns:
            Dictionary of evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Starting experiment: {self.config['experiment']['name']}")
        print(f"{'='*60}\n")

        # Save configuration for reproducibility
        config_save_path = Path(self.config['output']['save_dir']) / 'config.yaml'
        save_config(self.config, str(config_save_path))
        print(f"Saved configuration to {config_save_path}\n")

        # 1. Create dataloaders
        print("="*60)
        print("STEP 1: Loading Data")
        print("="*60)
        train_loader, test_loader, test_dataset = create_dataloaders(
            self.config['data']
        )

        # 2. Create encoder
        print(f"\n{'='*60}")
        print("STEP 2: Initializing Encoder")
        print("="*60)
        encoder = create_encoder(
            self.config['encoder'],
            device=self.device
        )
        print(f"Encoder: {self.config['encoder']['backbone']}")
        print(f"Layers: {self.config['encoder']['layers']}")
        print(f"Feature dimensions: {encoder.get_feature_dimensions()}")

        # 3. Extract or load cached training features
        print(f"\n{'='*60}")
        print("STEP 3: Extracting Training Features")
        print("="*60)
        cache = FeatureCache(self.config['cache'])

        if cache.has_cached_train_features():
            train_features = cache.load_train_features()
        else:
            train_features = encoder.extract_features(train_loader)
            cache.save_train_features(train_features)

        # 4. Create and fit algorithm
        print(f"\n{'='*60}")
        print("STEP 4: Fitting Anomaly Detection Algorithm")
        print("="*60)
        algorithm = create_algorithm(
            self.config['algorithm'],
            encoder_dims=encoder.get_feature_dimensions(),
            device=self.device
        )
        algorithm.fit(train_features)

        # Save fitted model
        model_path = Path(self.config['output']['model_dir']) / 'model.pkl'
        algorithm.save(str(model_path))
        print(f"\nSaved model to {model_path}")

        # 5. Extract test features and predict
        print(f"\n{'='*60}")
        print("STEP 5: Computing Anomaly Scores")
        print("="*60)
        test_features = encoder.extract_features(test_loader)
        anomaly_scores = algorithm.predict(test_features)

        # 6. Evaluate
        print(f"\n{'='*60}")
        print("STEP 6: Evaluating Performance")
        print("="*60)
        evaluator = Evaluator(self.config['evaluation'])
        results, gt_labels, img_scores, gt_masks = evaluator.evaluate(
            scores=anomaly_scores,
            test_dataset=test_dataset,
            test_loader=test_loader
        )

        # Print results
        if 'image_roc_auc' in results:
            print(f"Image-level ROC AUC: {results['image_roc_auc']:.3f}")
        if 'pixel_roc_auc' in results:
            print(f"Pixel-level ROC AUC: {results['pixel_roc_auc']:.3f}")
        if 'threshold' in results:
            print(f"Optimal threshold (F1-based): {results['threshold']:.3f}")

        # Save results (only summary metrics, not full ROC curve data)
        results_path = self.config['output']['results_file']
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_path}")

        # 7. Visualize
        if self.config['visualization']['enabled']:
            print(f"\n{'='*60}")
            print("STEP 7: Generating Visualizations")
            print("="*60)
            visualizer = AnomalyVisualizer(self.config['visualization'])
            visualizer.plot_results(
                scores=anomaly_scores,
                test_dataset=test_dataset,
                results=results,
                save_dir=self.config['output']['plots_dir'],
                gt_labels=gt_labels,
                img_scores=img_scores
            )

        print(f"\n{'='*60}")
        print("Pipeline Complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {self.config['output']['save_dir']}\n")

        return results
