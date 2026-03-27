import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def build_dataloaders_mimic(cfg: SimpleNamespace):
    path = str(cfg.mimic_preprocessed_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"MIMIC preprocessed path not found: {path}")

    df = pd.read_csv(path)
    label_col = str(cfg.label_col)
    
    # New: handle multiple sensitive columns
    sensitive_cols_str = getattr(cfg, "sensitive_cols", "minority")
    sensitive_cols = [c.strip() for c in sensitive_cols_str.split(",") if c.strip()]
    
    for c in [label_col] + sensitive_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in preprocessed MIMIC file.")

    exclude = {label_col} | set(sensitive_cols)
    if getattr(cfg, "id_cols", None):
        for c in str(cfg.id_cols).split(","):
            if c.strip():
                exclude.add(c.strip())

    feature_cols = getattr(cfg, "feature_cols", None)
    if feature_cols:
        feature_cols = [c.strip() for c in str(feature_cols).split(",") if c.strip()]
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if not feature_cols:
        raise ValueError("No numeric feature columns found. Please specify with --mimic-feature-cols.")

    df = df.dropna(subset=feature_cols + [label_col] + sensitive_cols).copy()

    y = df[label_col].astype(int).to_numpy()
    x = df[feature_cols].astype(np.float32).to_numpy()

    # Handle up to 3 sensitive attributes
    attrs = np.zeros((len(df), 3), dtype=np.int64)
    for i, col in enumerate(sensitive_cols[:3]): 
        attrs[:, i] = df[col].astype(int).to_numpy()

    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    x, y, attrs = x[idx], y[idx], attrs[idx]

    n_total = len(df)
    n_use = int(getattr(cfg, "n_use", 0) or 0)
    if 0 < n_use < n_total:
        n_total = n_use
        x, y, attrs = x[:n_total], y[:n_total], attrs[:n_total]

    train_frac = float(getattr(cfg, "train_frac", 0.6))
    cal_frac = float(getattr(cfg, "cal_frac", 0.2))
    train_end = max(1, min(int(np.floor(train_frac * n_total)), n_total - 2))
    cal_end = max(train_end + 1, min(int(np.floor((train_frac + cal_frac) * n_total)), n_total - 1))

    splits = {"train": slice(0, train_end), "cal": slice(train_end, cal_end), "test": slice(cal_end, n_total)}

    def make_loader(sl: slice, shuffle: bool):
        xb = torch.tensor(x[sl], dtype=torch.float32)
        yb = torch.tensor(y[sl], dtype=torch.long)
        a1 = torch.tensor(attrs[sl, 0], dtype=torch.long)
        a2 = torch.tensor(attrs[sl, 1], dtype=torch.long)
        a3 = torch.tensor(attrs[sl, 2], dtype=torch.long)
        ds = TensorDataset(xb, yb, a1, a2, a3)
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
        sensitive_cols=sensitive_cols,
        feature_dim=int(x.shape[1]),
    )
    return train_loader, cal_loader, test_loader, meta
