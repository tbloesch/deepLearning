"""Project-wide constants and settings."""
from pathlib import Path

import torch

# Dataset
DATA_FILE_ID = '1MK1FxwqvPrFvrtDsCccjSQIHWimpMWgG'   # Google Drive file id
DATA_DIR = Path('data')

# Preprocessing
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Loading
BATCH_SIZE = 32
NUM_WORKERS = 2


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device
