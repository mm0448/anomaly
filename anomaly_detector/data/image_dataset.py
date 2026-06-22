"""Image dataset with optional ground truth masks for anomaly detection."""

from pathlib import Path
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    Dataset for loading images with optional ground truth masks.

    Supports:
    - Loading images from a directory
    - Optional ground truth masks for anomalous samples
    - Flexible transforms for images and masks
    - Merging multiple datasets
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize image dataset.

        Args:
            image_dir: Directory containing images
            mask_dir: Optional directory containing ground truth masks
            transform: Torchvision transforms to apply to images
            mask_transform: Torchvision transforms to apply to masks
        """
        self.transform = transform
        self.mask_transform = mask_transform

        # Collect all image files
        self.image_files: List[Path] = [
            f for f in image_dir.glob("*")
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

        # Sort for deterministic ordering
        self.image_files.sort()

        # Find corresponding mask files if mask_dir is provided
        mask_files: List[Optional[Path]] = []
        missing: List[Path] = []

        for img_file in self.image_files:
            if mask_dir is None:
                mask_files.append(None)
            else:
                mask_path = mask_dir / f"{img_file.stem}_mask.png"
                if mask_path.exists():
                    mask_files.append(mask_path)
                else:
                    missing.append(mask_path)
                    mask_files.append(None)

        if missing:
            print(f"Warning: Missing {len(missing)} mask files")

        self.mask_files = mask_files

    def join(self, other: 'ImageDataset') -> None:
        """
        Merge another dataset into this one.

        Args:
            other: Another ImageDataset to merge
        """
        self.image_files = self.image_files + other.image_files
        if self.mask_files and other.mask_files:
            self.mask_files = self.mask_files + other.mask_files

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask at index.

        Args:
            idx: Index of sample

        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        # Load and transform image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load and transform mask (or create zero mask if none exists)
        mask_file = self.mask_files[idx]
        if mask_file:
            mask = Image.open(mask_file).convert('L')
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
        else:
            # Create zero mask with correct dimensions
            if self.mask_transform is not None:
                dummy_mask = Image.new('L', (224, 224), 0)
                mask = self.mask_transform(dummy_mask)
            else:
                mask = torch.zeros((1, image.shape[1], image.shape[2]))

        return image, mask
