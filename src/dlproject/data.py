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


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_datasets(data_dir=DATA_DIR, transform=None):
    """Return (train_dataset, val_dataset) as ImageFolder datasets."""
    data_dir = Path(data_dir)
    transform = transform or get_transform()
    train_dataset = datasets.ImageFolder(data_dir / 'train', transform=transform)
    val_dataset   = datasets.ImageFolder(data_dir / 'val',   transform=transform)
    return train_dataset, val_dataset


def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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
