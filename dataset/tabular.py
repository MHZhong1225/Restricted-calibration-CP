import csv
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, x, y, attrs: Sequence[np.ndarray]):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.attrs = [torch.tensor(a, dtype=torch.long) for a in attrs]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx], *[a[idx] for a in self.attrs])


def _split_csv_list(value: str) -> List[str]:
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def _read_csv_rows(path: str, fieldnames=None):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Provide the path with the corresponding CLI flag."
        )
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Dataset file is empty: {path}")
    return rows, reader.fieldnames or []


def _encode_column(rows, name: str):
    values = [row.get(name, "") for row in rows]
    out = []
    mapping: Dict[str, int] = {}
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            if value not in mapping:
                mapping[value] = len(mapping)
            out.append(float(mapping[value]))
    return np.asarray(out, dtype=np.float32)


def _encode_label(rows, name: str):
    raw = [row.get(name, "") for row in rows]
    mapping: Dict[str, int] = {}
    labels = []
    for value in raw:
        try:
            labels.append(int(float(value)))
        except (TypeError, ValueError):
            if value not in mapping:
                mapping[value] = len(mapping)
            labels.append(mapping[value])
    return np.asarray(labels, dtype=np.int64)


def _choose_feature_cols(all_cols, label_col, sensitive_cols, requested):
    requested_cols = _split_csv_list(requested)
    if requested_cols:
        missing = [c for c in requested_cols if c not in all_cols]
        if missing:
            raise ValueError(f"Missing requested feature columns: {missing}")
        return requested_cols
    blocked = {label_col, *sensitive_cols}
    return [c for c in all_cols if c not in blocked]


def _build_arrays(rows, all_cols, label_col, feature_cols, attr_cols):
    missing = [c for c in [label_col, *feature_cols, *attr_cols] if c not in all_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    x = np.stack([_encode_column(rows, c) for c in feature_cols], axis=1).astype(np.float32)
    y = _encode_label(rows, label_col)
    attrs = [_encode_column(rows, c).astype(np.int64) for c in attr_cols]
    return x, y, attrs


def _normalize_features(x):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.clip(sd, 1e-6, None)


def _split_indices(n, seed, n_use, train_frac, test_samples):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    if int(n_use) > 0:
        perm = perm[: min(int(n_use), n)]

    if len(perm) < 3:
        raise ValueError(f"Need at least 3 rows after n_use filtering, got {len(perm)}")

    n_test = int(test_samples or 0)
    if n_test <= 0 or n_test >= len(perm) - 2:
        n_test = max(1, int(round(0.2 * len(perm))))
    test_idx = perm[:n_test]
    train_cal_idx = perm[n_test:]

    n_train = int(round(len(train_cal_idx) * float(train_frac)))
    n_train = min(max(n_train, 1), len(train_cal_idx) - 1)
    return train_cal_idx[:n_train], train_cal_idx[n_train:], test_idx


def _split_indices_by_fracs(n, seed, n_use, train_frac, cal_frac):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    if int(n_use) > 0:
        perm = perm[: min(int(n_use), n)]

    if len(perm) < 3:
        raise ValueError(f"Need at least 3 rows after n_use filtering, got {len(perm)}")

    n_train = int(round(len(perm) * float(train_frac)))
    n_cal = int(round(len(perm) * float(cal_frac)))
    n_train = max(n_train, 1)
    n_cal = max(n_cal, 1)
    if n_train + n_cal >= len(perm):
        n_cal = max(1, len(perm) - n_train - 1)
    return perm[:n_train], perm[n_train : n_train + n_cal], perm[n_train + n_cal :]


def _make_loaders(x, y, attrs, cfg):
    x = _normalize_features(x)
    tr, cal, test = _split_indices(
        len(y),
        seed=cfg.get("seed", 42),
        n_use=cfg.get("n_use", 0),
        train_frac=cfg.get("train_frac", 0.5),
        test_samples=cfg.get("test_samples", 0),
    )
    batch_size = int(cfg.get("batch_size", 128))

    def ds(indices):
        return TabularDataset(x[indices], y[indices], [a[indices] for a in attrs])

    return (
        DataLoader(ds(tr), batch_size=batch_size, shuffle=True),
        DataLoader(ds(cal), batch_size=batch_size, shuffle=False),
        DataLoader(ds(test), batch_size=batch_size, shuffle=False),
    )


def _make_mimic_loaders(x, y, attrs, cfg):
    x = _normalize_features(x)
    tr, cal, test = _split_indices_by_fracs(
        len(y),
        seed=cfg.get("seed", 42),
        n_use=cfg.get("n_use", 0),
        train_frac=cfg.get("mimic_train_frac", 0.6),
        cal_frac=cfg.get("mimic_cal_frac", 0.2),
    )
    batch_size = int(cfg.get("batch_size", 128))

    def ds(indices):
        return TabularDataset(x[indices], y[indices], [a[indices] for a in attrs])

    return (
        DataLoader(ds(tr), batch_size=batch_size, shuffle=True),
        DataLoader(ds(cal), batch_size=batch_size, shuffle=False),
        DataLoader(ds(test), batch_size=batch_size, shuffle=False),
    )


def build_dataloaders_mimic(cfg):
    rows, cols = _read_csv_rows(cfg.get("mimic_preprocessed_path", "dataset/mimic-iv_processed.csv"))
    label_col = cfg.get("mimic_label_col", "label")
    attr_cols = _split_csv_list(cfg.get("mimic_sensitive_cols", "minority,gender_m,public_insurance"))
    feature_cols = _choose_feature_cols(cols, label_col, attr_cols, cfg.get("mimic_feature_cols", ""))
    x, y, attrs = _build_arrays(rows, cols, label_col, feature_cols, attr_cols)
    return _make_mimic_loaders(x, y, attrs, cfg)


def build_dataloaders_nursery(cfg):
    path = cfg.get("nursery_csv_path", "dataset/nursery/nursery.csv")
    with open(path, "r", newline="", encoding="utf-8") as f:
        first = next(csv.reader(f))
    expected_cols = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
    has_header = any(col in first for col in expected_cols)
    rows, cols = _read_csv_rows(path, fieldnames=None if has_header else expected_cols)
    label_col = "class" if "class" in cols else cols[-1]
    attr_cols = [c for c in ["parents", "children", "finance", "social", "health"] if c in cols]
    feature_cols = [c for c in cols if c != label_col]
    x, y, attrs = _build_arrays(rows, cols, label_col, feature_cols, attr_cols)
    return _make_loaders(x, y, attrs, cfg)
