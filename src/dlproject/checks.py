"""Sanity checks to run on the data before training."""
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .utils import dataset_labels, denormalise, list_images


def visualise_samples(dataset, class_names, n_per_class=4):
    """
    Show n_per_class examples from each class.
    The most important sanity check — do the images and labels look right?
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 3, n_classes * 3),
        squeeze=False,
    )

    # Collect indices per class
    class_indices = {c: [] for c in range(n_classes)}
    for idx, label in enumerate(dataset_labels(dataset)):
        if len(class_indices[label]) < n_per_class:
            class_indices[label].append(idx)
        if all(len(v) >= n_per_class for v in class_indices.values()):
            break

    for row, (cls, indices) in enumerate(class_indices.items()):
        for col, idx in enumerate(indices[:n_per_class]):
            ax  = axes[row][col]
            img = denormalise(dataset[idx][0]).permute(1, 2, 0).numpy()
            ax.imshow(img)
            # Hide ticks and frame but keep the y-label (axis('off') would hide it too)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(class_names[cls], fontsize=11, fontweight='bold')

    plt.suptitle('Sample images per class — check that labels are correct', fontsize=12)
    plt.tight_layout()
    plt.show()


def check_class_distribution(dataset, class_names, split_name='dataset'):
    """
    Plot the number of images per class.
    Large imbalances will bias the model towards the majority class.
    """
    counts = Counter(dataset_labels(dataset))
    names  = [class_names[i] for i in sorted(counts.keys())]
    values = [counts[i] for i in sorted(counts.keys())]
    total  = sum(values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    bars = ax1.bar(names, values, color='steelblue', edgecolor='white')
    ax1.set_title(f'Class distribution — {split_name}')
    ax1.set_ylabel('Number of images')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=10)

    # Pie chart showing balance
    ax2.pie(values, labels=names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class balance')

    plt.tight_layout()
    plt.show()

    print(f'Total: {total} images')
    print(f'Min class: {min(values)} ({names[values.index(min(values))]})')
    print(f'Max class: {max(values)} ({names[values.index(max(values))]})')
    ratio = max(values) / max(min(values), 1)
    if ratio > 3:
        print(f'WARNING: Class imbalance ratio {ratio:.1f}x — consider resampling or weighted loss.')
    else:
        print(f'Class imbalance ratio: {ratio:.1f}x — looks balanced.')


def check_image_sizes(image_dir, sample_size=200):
    """
    Check the distribution of original image sizes before resizing.
    Extreme size variation can indicate mixed data sources or quality issues.
    """
    widths, heights = [], []
    image_files = list_images(image_dir)

    # Sample a subset for speed
    sampled = random.sample(image_files, min(sample_size, len(image_files)))

    for path in sampled:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass   # skip unreadable files

    if not widths:
        print(f'No readable images found in {image_dir}.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(widths,  bins=20, color='steelblue', edgecolor='white')
    axes[0].set_title('Image widths'); axes[0].set_xlabel('Pixels')

    axes[1].hist(heights, bins=20, color='coral',     edgecolor='white')
    axes[1].set_title('Image heights'); axes[1].set_xlabel('Pixels')

    plt.suptitle(f'Image size distribution (sample of {len(widths)})', fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f'Width:  min={min(widths)}, max={max(widths)}, median={int(np.median(widths))}')
    print(f'Height: min={min(heights)}, max={max(heights)}, median={int(np.median(heights))}')

    if max(widths) / max(min(widths), 1) > 10:
        print('WARNING: Very large variation in image sizes — check for outliers.')


def check_tensor_stats(loader, n_batches=5):
    """
    Check the mean and std of pixel values after normalisation.
    After ImageNet normalisation, mean should be near 0 and std near 1.
    Large deviations suggest the wrong normalisation was applied.
    """
    all_means = []
    all_stds  = []

    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        all_means.append(images.mean(dim=[0, 2, 3]).numpy())  # per channel
        all_stds.append(images.std(dim=[0, 2, 3]).numpy())

    mean = np.array(all_means).mean(axis=0)
    std  = np.array(all_stds).mean(axis=0)

    print('After normalisation (should be near mean≈0, std≈1 per channel):')
    for c, name in enumerate(['R', 'G', 'B']):
        print(f'  {name}: mean={mean[c]:.3f}, std={std[c]:.3f}')

    if abs(mean).max() > 0.5:
        print('WARNING: Mean far from 0 — check normalisation values.')
    if abs(std - 1).max() > 0.5:
        print('WARNING: Std far from 1 — check normalisation values.')


def check_split(train_dataset, val_dataset, class_names):
    """
    Verify the train/val split looks reasonable.
    The split should be consistent across classes.
    """
    train_counts = Counter(dataset_labels(train_dataset))
    val_counts   = Counter(dataset_labels(val_dataset))

    print(f"{'Class':<20} {'Train':>8} {'Val':>8} {'Val %':>8}")
    print('-' * 46)
    for i, name in enumerate(class_names):
        tr  = train_counts[i]
        vl  = val_counts[i]
        pct = vl / (tr + vl) * 100 if (tr + vl) > 0 else 0
        print(f'{name:<20} {tr:>8} {vl:>8} {pct:>7.1f}%')

    total_train = sum(train_counts.values())
    total_val   = sum(val_counts.values())
    total_pct   = total_val / (total_train + total_val) * 100
    print('-' * 46)
    print(f"{'TOTAL':<20} {total_train:>8} {total_val:>8} {total_pct:>7.1f}%")

    if total_pct < 10:
        print('WARNING: Validation set is very small — results may be noisy.')
    if total_pct > 40:
        print('WARNING: Validation set is large — consider using more data for training.')


def find_corrupt_images(image_dir):
    """
    Scan for images that cannot be opened.
    Corrupt images will crash training with cryptic errors.
    Run this once before you start training.
    """
    image_files = list_images(image_dir)
    corrupt = []
    for path in image_files:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            corrupt.append((path, str(e)))

    if corrupt:
        print(f'Found {len(corrupt)} corrupt image(s):')
        for path, err in corrupt:
            print(f'  {path}: {err}')
    else:
        print(f'Scanned {len(image_files)} images — no corrupt files found.')
    return corrupt


def find_train_val_duplicates(train_dataset, val_dataset, device=None,
                              threshold=0.95, batch_size=32):
    """
    Find likely duplicate images across training and validation datasets.

    Why? A validation image that also appears in the training set makes
    validation accuracy misleading.

    How? A pretrained ResNet-18 extracts one feature vector per image. Pairs
    with cosine similarity at or above `threshold` are returned for manual
    review. A high threshold is intentionally used because similar is not
    necessarily identical: augmentation, compression, or resizing can change
    pixel values.

    Note that this is just one type of train/val data leakage — there are many
    other ways to leak information. For example, if the validation set contains
    images of the same objects (e.g., patients) as the training set, the model
    may perform well on validation but fail to generalize to new objects.
    """
    from torchvision.models import resnet18, ResNet18_Weights
    import torch.nn.functional as F

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def dataset_paths(dataset):
        if hasattr(dataset, 'samples'):
            return [path for path, _ in dataset.samples]
        if hasattr(dataset, 'imgs'):
            return [path for path, _ in dataset.imgs]
        raise TypeError("Expected an ImageFolder-style dataset with `.samples` or `.imgs`.")

    train_paths = dataset_paths(train_dataset)
    val_paths = dataset_paths(val_dataset)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    def extract_features(dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        features = []
        with torch.inference_mode():
            for images, _ in loader:
                batch_features = model(images.to(device))
                features.append(F.normalize(batch_features, p=2, dim=1).cpu())
        return torch.cat(features)

    train_features = extract_features(train_dataset)
    val_features = extract_features(val_dataset)
    similarity = val_features @ train_features.T
    val_indices, train_indices = torch.where(similarity >= threshold)

    pairs = []
    for val_index, train_index in zip(val_indices.tolist(), train_indices.tolist()):
        pairs.append({
            'train_path': train_paths[train_index],
            'val_path': val_paths[val_index],
            'similarity': float(similarity[val_index, train_index]),
        })

    pairs.sort(key=lambda pair: pair['similarity'], reverse=True)
    print(f'Found {len(pairs)} possible train/validation duplicate pair(s).')
    for pair in pairs[:10]:
        print(f"{pair['similarity']:.3f}  train: {pair['train_path']}  val: {pair['val_path']}")
    return pairs
