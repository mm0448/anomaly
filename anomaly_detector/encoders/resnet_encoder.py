"""ResNet-based feature encoder for anomaly detection."""

from collections import OrderedDict
from typing import Dict, List
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from tqdm import tqdm

from ..core.base_encoder import BaseEncoder


class ResNetEncoder(BaseEncoder):
    """
    ResNet-based feature encoder for anomaly detection.

    Extracts multi-scale features from intermediate ResNet layers.
    Supports ResNet18, ResNet34, ResNet50, and ResNet101 variants.
    """

    LAYER_DIMS = {
        'resnet18': {'layer1': 64, 'layer2': 128, 'layer3': 256},
        'resnet34': {'layer1': 64, 'layer2': 128, 'layer3': 256},
        'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024},
        'resnet101': {'layer1': 256, 'layer2': 512, 'layer3': 1024},
    }

    def __init__(
        self,
        backbone: str = 'resnet18',
        layers: List[str] = None,
        pretrained: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize ResNet encoder.

        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            layers: Layers to extract features from (default: ['layer1', 'layer2', 'layer3'])
            pretrained: Use ImageNet pretrained weights
            device: Device to run on ('cpu' or 'cuda:0')
        """
        if layers is None:
            layers = ['layer1', 'layer2', 'layer3']

        super().__init__(layers, device)
        self.backbone = backbone
        self.pretrained = pretrained
        self.outputs = OrderedDict()
        self.model = self.build_model()
        self._register_hooks()

    def build_model(self) -> nn.Module:
        """Build ResNet model."""
        model_fn = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
        }

        if self.backbone not in model_fn:
            raise ValueError(
                f"Unknown backbone: {self.backbone}. "
                f"Choose from: {list(model_fn.keys())}"
            )

        model = model_fn[self.backbone](pretrained=self.pretrained)
        model.to(self.device)
        model.eval()
        return model

    def _register_hooks(self):
        """Register forward hooks for feature extraction."""
        def get_hook(name):
            def hook(module, input, output):
                self.outputs[name] = output
            return hook

        for layer_name in self.layers:
            if not hasattr(self.model, layer_name):
                raise ValueError(
                    f"Layer {layer_name} not found in {self.backbone}"
                )

            layer = getattr(self.model, layer_name)
            # Hook to the last block of each layer
            hook = layer[-1].register_forward_hook(get_hook(layer_name))
            self.hooks.append(hook)

    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get feature dimensions for each layer.

        Returns:
            Dictionary mapping layer names to feature channel dimensions
        """
        return {
            layer: self.LAYER_DIMS[self.backbone][layer]
            for layer in self.layers
        }

    def extract_features(self, dataloader) -> Dict[str, torch.Tensor]:
        """
        Extract features from all images in dataloader.

        Args:
            dataloader: PyTorch DataLoader yielding (image, mask) tuples

        Returns:
            Dictionary mapping layer names to feature tensors of shape (N, C, H, W)
        """
        feature_dict = OrderedDict({layer: [] for layer in self.layers})

        with torch.no_grad():
            for x, _ in tqdm(dataloader, desc='Extracting features'):
                self.outputs.clear()
                _ = self.model(x.to(self.device))

                for layer_name in self.layers:
                    feature_dict[layer_name].append(
                        self.outputs[layer_name].cpu().detach()
                    )

        # Concatenate batches
        for layer_name in self.layers:
            feature_dict[layer_name] = torch.cat(feature_dict[layer_name], 0)

        return feature_dict
