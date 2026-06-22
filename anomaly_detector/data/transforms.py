"""Standard image transforms for anomaly detection."""

import torchvision.transforms as transforms
from typing import Dict, Tuple


def get_standard_transforms(
    config: Dict
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create standard image and mask transforms from configuration.

    Args:
        config: Transform configuration dictionary with keys:
            - resize: Target size for resizing
            - center_crop: Size for center cropping
            - normalize: Dict with 'mean' and 'std' lists

    Returns:
        Tuple of (image_transform, mask_transform)
    """
    transform_config = config.get('transforms', {})
    resize = transform_config.get('resize', 256)
    center_crop = transform_config.get('center_crop', 224)
    normalize = transform_config.get('normalize', {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    })

    # Image transform with normalization
    image_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalize['mean'],
            std=normalize['std']
        )
    ])

    # Mask transform (no normalization for binary masks)
    mask_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor()
    ])

    return image_transform, mask_transform
