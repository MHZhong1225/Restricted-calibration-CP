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

def build_dataloaders_adult(cfg: SimpleNamespace):
    path = str(getattr(cfg, "adult_csv_path", "adult.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"adult_csv_path not found: {path}")

    df = pd.read_csv(path)

    y = (df["income"] == ">50K").astype(np.int32).to_numpy()

    # 2. 敏感属性处理：race 性别 
    # 默认：color = 1 如果是 Minority (非 White)，color = 0 如果是 White
    color = (df["race"] != "White").astype(np.int32).to_numpy()
    
    # 3. 年龄处理：将年龄分桶 (例如 0-30, 31-45, ...)
    age_bucket = _bucketize_age(df["age"].to_numpy())
    
    # 4. 额外分组 (region): 这里我们使用 sex，Male=0, Female=1
    region = (df["sex"] == "Female").astype(np.int32).to_numpy()

    # 5. 提取特征并处理类别变量
    # 排除直接作为 target 或 group 的列
    exclude_cols = ["income", "race", "sex", "age"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 对类别特征进行 one-hot 编码
    df_features = pd.get_dummies(df[feature_cols], drop_first=True)
    # 标准化数值特征
    for c in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[c]) and df_features[c].nunique() > 2:
            mean = df_features[c].mean()
            std = df_features[c].std()
            if std > 0:
                df_features[c] = (df_features[c] - mean) / std
            else:
                df_features[c] = 0.0
                
    x = df_features.astype(np.float32).to_numpy()

    # 6. 数据打乱和划分
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
        # For compatibility with eval code that expects diag
        db = torch.zeros_like(yb)
        ds = TensorDataset(xb, yb, cb, ab, rb, db, torch.zeros_like(yb))
        return DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=shuffle, drop_last=False)

    train_loader = make_loader(splits["train"], shuffle=True)
    cal_loader = make_loader(splits["cal"], shuffle=False)
    test_loader = make_loader(splits["test"], shuffle=False)

    meta = SimpleNamespace(
        dataset_mode="adult",
        path=path,
        n_total=n_total,
        n_train=train_end,
        n_cal=cal_end - train_end,
        n_test=n_total - cal_end,
        feature_dim=int(x.shape[1]),
    )
    return train_loader, cal_loader, test_loader, meta
