import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def _bucketize_age(age_years: np.ndarray) -> np.ndarray:
    age_years = np.asarray(age_years, dtype=np.float32)
    bins = np.asarray([0, 30, 45, 60, 75, 200], dtype=np.float32)
    return np.digitize(age_years, bins[1:-1], right=False).astype(np.int64) + 1


def build_dataloaders_mimic(cfg: SimpleNamespace):
    path = str(cfg.mimic_preprocessed_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"mimic_preprocessed_path not found: {path}")

    df = pd.read_csv(path)

    label_col = str(cfg.label_col)
    sensitive_col = str(cfg.sensitive_col)
    age_col = str(cfg.age_col)
    region_col = str(cfg.region_col) if getattr(cfg, "region_col", None) else None

    for c in [label_col, sensitive_col, age_col]:
        if c not in df.columns:
            raise ValueError(f"Column not found in preprocessed MIMIC file: {c}")

    if region_col is not None and region_col not in df.columns:
        raise ValueError(f"region_col not found in preprocessed MIMIC file: {region_col}")

    exclude = {label_col, sensitive_col, age_col}
    if region_col is not None:
        exclude.add(region_col)
    if getattr(cfg, "id_cols", None):
        for c in str(cfg.id_cols).split(","):
            c = c.strip()
            if c:
                exclude.add(c)

    feature_cols = getattr(cfg, "feature_cols", None)
    if feature_cols:
        feature_cols = [c.strip() for c in str(feature_cols).split(",") if c.strip()]
    else:
        feature_cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found. Provide --mimic-feature-cols.")

    df = df.dropna(subset=feature_cols + [label_col, sensitive_col, age_col]).copy()

    y = df[label_col].astype(int).to_numpy()
    color = df[sensitive_col].astype(int).to_numpy()
    age_bucket = _bucketize_age(df[age_col].astype(float).to_numpy())
    region = np.zeros(len(df), dtype=np.int64) if region_col is None else df[region_col].astype(int).to_numpy()
    x = df[feature_cols].astype(np.float32).to_numpy()

    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    x, y, color, age_bucket, region = x[idx], y[idx], color[idx], age_bucket[idx], region[idx]

    n_total = len(df)
    n_use = int(getattr(cfg, "n_use", 0) or 0)
    if n_use > 0 and n_use < n_total:
        n_total = n_use
        x, y, color, age_bucket, region = x[:n_total], y[:n_total], color[:n_total], age_bucket[:n_total], region[:n_total]

    train_frac = float(getattr(cfg, "train_frac", 0.6))
    cal_frac = float(getattr(cfg, "cal_frac", 0.2))
    train_end = int(np.floor(train_frac * n_total))
    cal_end = int(np.floor((train_frac + cal_frac) * n_total))
    train_end = max(1, min(train_end, n_total - 2))
    cal_end = max(train_end + 1, min(cal_end, n_total - 1))

    splits = {
        "train": slice(0, train_end),
        "cal": slice(train_end, cal_end),
        "test": slice(cal_end, n_total),
    }

    def make_loader(sl: slice, shuffle: bool):
        xb = torch.tensor(x[sl], dtype=torch.float32)
        yb = torch.tensor(y[sl], dtype=torch.long)
        cb = torch.tensor(color[sl], dtype=torch.long)
        ab = torch.tensor(age_bucket[sl], dtype=torch.long)
        rb = torch.tensor(region[sl], dtype=torch.long)
        ds = TensorDataset(xb, yb, cb, ab, rb)
        return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=shuffle, drop_last=False)

    train_loader = make_loader(splits["train"], shuffle=True)
    cal_loader = make_loader(splits["cal"], shuffle=False)
    test_loader = make_loader(splits["test"], shuffle=False)

    meta = SimpleNamespace(
        dataset_mode="mimic",
        path=path,
        n_total=n_total,
        n_train=train_end,
        n_cal=cal_end - train_end,
        n_test=n_total - cal_end,
        label_col=label_col,
        sensitive_col=sensitive_col,
        age_col=age_col,
        region_col=region_col,
        feature_dim=int(x.shape[1]),
    )
    return train_loader, cal_loader, test_loader, meta

