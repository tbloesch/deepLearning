"""Small helpers shared across modules."""
from pathlib import Path

import torch

from .config import IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def denormalise(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def list_images(image_dir):
    """All image files below image_dir (case-insensitive extension match)."""
    return [p for p in Path(image_dir).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]


def dataset_labels(dataset):
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    return [dataset[i][1] for i in range(len(dataset))]
