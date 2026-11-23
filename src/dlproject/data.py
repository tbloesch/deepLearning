"""Downloading the dataset and building datasets / dataloaders."""
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import (BATCH_SIZE, DATA_DIR, DATA_FILE_ID, IMAGE_SIZE,
                     IMAGENET_MEAN, IMAGENET_STD, NUM_WORKERS)


def download_data(data_dir=DATA_DIR, file_id=DATA_FILE_ID):
    """Download and extract the dataset from Google Drive (skipped if it already exists)."""
    data_dir = Path(data_dir)
    if data_dir.exists():
        print(f'Dataset already present at {data_dir}/')
        return data_dir

    import gdown
    zip_path = 'data.zip'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir.parent)
    print('Dataset ready.')
    return data_dir


def get_transform(flip=False, crop=False, jitter=False):
    """Return a composed transform with optional random crop, flip, and color jitter."""
    steps = []
    
    if crop:
        steps.append(transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)))
    else:
        steps.append(transforms.Resize(IMAGE_SIZE))

    if flip:
        steps.append(transforms.RandomHorizontalFlip())
    if jitter:
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


def get_datasets(data_dir=DATA_DIR, train_transform=None, val_transform=None):
    """Return (train_dataset, val_dataset) as ImageFolder datasets."""
    data_dir = Path(data_dir)
    train_transform = train_transform or get_transform()
    val_transform = val_transform or get_transform()
    train_dataset = datasets.ImageFolder(data_dir / 'train', transform=train_transform)
    val_dataset   = datasets.ImageFolder(data_dir / 'val',   transform=val_transform)
    return train_dataset, val_dataset


def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

def get_test_dataset(data_dir=DATA_DIR, test_transform=None):
    """Return the test dataset as an ImageFolder dataset."""
    data_dir = Path(data_dir)
    test_transform = test_transform or get_transform()
    test_dataset = datasets.ImageFolder(data_dir / 'test', transform=test_transform)
    return test_dataset

def describe_dataset(dataset):
    """Print classes, size and the shape of the first item."""
    print(f'Classes:          {dataset.classes}')
    print(f'Class to index:   {dataset.class_to_idx}')
    print(f'Number of images: {len(dataset)}')
    image, label = dataset[0]
    print('\nFirst item:')
    print(f'  Image shape: {image.shape}')   # (C, H, W)
    print(f'  Label:       {label} ({dataset.classes[label]})')


def describe_batch(loader):
    images, labels = next(iter(loader))
    print(f'Batch image shape: {images.shape}')   # (B, 3, H, W)
    print(f'Batch label shape: {labels.shape}')   # (B,)
    print(f'Labels in batch:   {labels.tolist()}')
