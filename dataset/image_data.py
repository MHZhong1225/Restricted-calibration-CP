import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


BACH_CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]
BACH_EXTS = ["tif", "tiff", "png", "jpg", "jpeg"]


def collect_bach_paths(source_dir: Path, classes=None, exts=None):
    classes = classes or BACH_CLASSES
    exts = exts or BACH_EXTS
    data = []
    for class_idx, class_name in enumerate(classes):
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            continue
        class_paths = []
        for ext in exts:
            class_paths.extend(glob.glob(str(class_dir / f"*.{ext}")))
        for path in sorted(set(class_paths)):
            data.append((path, class_idx))
    return data


class BachDataset(Dataset):
    def __init__(self, samples, transform):
        if len(samples) == 0:
            raise ValueError("BACH dataset received no image samples.")
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path, label = self.samples[idx]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        y = torch.tensor(label, dtype=torch.long)
        return x, y, y


def _stratified_split(samples, train_frac, test_samples, seed):
    from sklearn.model_selection import train_test_split

    if len(samples) < 3:
        raise ValueError(f"Need at least 3 BACH images, got {len(samples)}")

    labels = [label for _path, label in samples]
    n_test = int(test_samples or 0)
    if n_test <= 0 or n_test >= len(samples) - 2:
        n_test = max(1, int(round(0.2 * len(samples))))
    if n_test < len(set(labels)):
        n_test = len(set(labels))

    train_cal, test = train_test_split(
        samples,
        test_size=n_test,
        random_state=int(seed),
        stratify=labels,
    )
    train_cal_labels = [label for _path, label in train_cal]
    n_train = int(round(len(train_cal) * float(train_frac)))
    n_train = min(max(n_train, len(set(train_cal_labels))), len(train_cal) - len(set(train_cal_labels)))
    train, cal = train_test_split(
        train_cal,
        train_size=n_train,
        random_state=int(seed),
        stratify=train_cal_labels,
    )
    return train, cal, test


def _maybe_subset(samples, n_use, seed):
    n_use = int(n_use or 0)
    if n_use <= 0 or n_use >= len(samples):
        return samples
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(samples))[:n_use]
    return [samples[i] for i in idx]


def _load_split_layout(root: Path):
    train = collect_bach_paths(root / "train")
    cal = collect_bach_paths(root / "val")
    test = collect_bach_paths(root / "test")
    if len(train) == 0 or len(cal) == 0 or len(test) == 0:
        return None
    return train, cal, test


def _load_flat_layout(root: Path, cfg):
    samples = collect_bach_paths(root)
    if len(samples) == 0:
        raise RuntimeError(f"No BACH images found under {root} for classes={BACH_CLASSES} exts={BACH_EXTS}")
    samples = _maybe_subset(samples, cfg.get("n_use", 0), cfg.get("seed", 42))
    return _stratified_split(
        samples,
        train_frac=cfg.get("train_frac", 0.5),
        test_samples=cfg.get("test_samples", 100),
        seed=cfg.get("seed", 42),
    )


def build_dataloaders_bach(cfg):
    from torchvision import transforms

    root = Path(cfg.get("bach_data_dir", "dataset/datasets/bach")).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"BACH data directory not found: {root}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split_samples = _load_split_layout(root)
    if split_samples is None:
        split_samples = _load_flat_layout(root, cfg)
    train, cal, test = split_samples

    batch_size = int(cfg.get("batch_size", 32))
    return (
        DataLoader(BachDataset(train, transform), batch_size=batch_size, shuffle=True),
        DataLoader(BachDataset(cal, transform), batch_size=batch_size, shuffle=False),
        DataLoader(BachDataset(test, transform), batch_size=batch_size, shuffle=False),
    )
