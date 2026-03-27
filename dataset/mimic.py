import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def build_dataloaders_mimic(cfg: SimpleNamespace):
    path = str(cfg.mimic_preprocessed_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"mimic_preprocessed_path not found: {path}")

    df = pd.read_csv(path)
    label_col = str(cfg.label_col)
    
    # 传入的多个敏感属性列（逗号分隔）
    sens_cols = [c.strip() for c in str(getattr(cfg, "sensitive_cols", "minority")).split(",") if c.strip()]
    
    for c in [label_col] + sens_cols:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    exclude = {label_col} | set(sens_cols)
    if getattr(cfg, "id_cols", None):
        for c in str(cfg.id_cols).split(","):
            exclude.add(c.strip())

    feature_cols = getattr(cfg, "feature_cols", None)
    if feature_cols:
        feature_cols = [c.strip() for c in str(feature_cols).split(",") if c.strip()]
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    df = df.dropna(subset=feature_cols + [label_col] + sens_cols).copy()

    y = df[label_col].astype(int).to_numpy()
    x = df[feature_cols].astype(np.float32).to_numpy()

    # 最多3个敏感属性
    attrs = np.zeros((len(df), 3), dtype=np.int64)
    for i, col in enumerate(sens_cols[:3]): 
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
        return DataLoader(TensorDataset(xb, yb, a1, a2, a3), batch_size=int(cfg.batch_size), shuffle=shuffle)

    meta = SimpleNamespace(dataset_mode="mimic", n_total=n_total, feature_dim=int(x.shape[1]))
    return make_loader(splits["train"], True), make_loader(splits["cal"], False), make_loader(splits["test"], False), meta