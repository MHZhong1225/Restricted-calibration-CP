import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def build_dataloaders_nursery(cfg: SimpleNamespace):
    path = str(getattr(cfg, "nursery_csv_path", "./nursery/nursery.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"nursery_csv_path not found:s {path}")

    df = pd.read_csv(path)

    # Label: "final evaluation" -> 0-4
    label_map = {
        "not_recom": 0,
        "recommend": 1,
        "very_recom": 2,
        "priority": 3,
        "spec_prior": 4
    }
    y = df["final evaluation"].map(label_map).astype(np.int64).to_numpy()

    # Features: one-hot encoding all features
    feature_cols = [c for c in df.columns if c != "final evaluation"]
    df_features = pd.get_dummies(df[feature_cols], drop_first=True)
    x = df_features.astype(np.float32).to_numpy()

    # Sensitive attributes: all except housing
    # Categorical encoding to integer labels for each sensitive attribute
    attr1 = df["parents"].astype("category").cat.codes.to_numpy() # 3 levels
    attr2 = df["has_nurs"].astype("category").cat.codes.to_numpy() # 5 levels
    attr3 = df["form"].astype("category").cat.codes.to_numpy() # 4 levels
    attr4 = df["children"].astype("category").cat.codes.to_numpy() # 4 levels
    attr5 = df["finance"].astype("category").cat.codes.to_numpy() # 2 levels
    attr6 = df["social"].astype("category").cat.codes.to_numpy() # 3 levels
    attr7 = df["health"].astype("category").cat.codes.to_numpy() # 3 levels

    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    x = x[idx]
    y = y[idx]
    attr1, attr2, attr3 = attr1[idx], attr2[idx], attr3[idx]
    attr4, attr5, attr6, attr7 = attr4[idx], attr5[idx], attr6[idx], attr7[idx]

    n_total = len(df)
    n_use = int(getattr(cfg, "n_use", 0) or 0)
    if n_use > 0 and n_use < n_total:
        n_total = n_use
        x = x[:n_total]
        y = y[:n_total]
        attr1, attr2, attr3 = attr1[:n_total], attr2[:n_total], attr3[:n_total]
        attr4, attr5, attr6, attr7 = attr4[:n_total], attr5[:n_total], attr6[:n_total], attr7[:n_total]

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
        a1 = torch.tensor(attr1[sl], dtype=torch.long)
        a2 = torch.tensor(attr2[sl], dtype=torch.long)
        a3 = torch.tensor(attr3[sl], dtype=torch.long)
        a4 = torch.tensor(attr4[sl], dtype=torch.long)
        a5 = torch.tensor(attr5[sl], dtype=torch.long)
        a6 = torch.tensor(attr6[sl], dtype=torch.long)
        a7 = torch.tensor(attr7[sl], dtype=torch.long)
        
        # We need a 9-element batch (x, y, a1..a7)
        ds = TensorDataset(xb, yb, a1, a2, a3, a4, a5, a6, a7)
        return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=shuffle, drop_last=False)

    train_loader = make_loader(splits["train"], shuffle=True)
    cal_loader = make_loader(splits["cal"], shuffle=False)
    test_loader = make_loader(splits["test"], shuffle=False)

    meta = SimpleNamespace(
        dataset_mode="nursery",
        path=path,
        n_total=n_total,
        n_train=train_end,
        n_cal=cal_end - train_end,
        n_test=n_total - cal_end,
        feature_dim=int(x.shape[1]),
    )
    return train_loader, cal_loader, test_loader, meta
