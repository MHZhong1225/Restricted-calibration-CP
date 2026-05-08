from itertools import islice, cycle
import numpy as np
import torch
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Data generating model with inside-group variance
class data_model:
    def __init__(
        self,
        p,
        delta1,
        delta0,
        group_perc=[0.5, 0.5],
        age_perc=None,
        region_perc=None,
        K=6,
        seed=2026,
        sex_prob=0.5,
        pheno_prob=0.5,
        n_nonsensitive=6,
        **kwargs,
    ):
        self.K = K
        self.p = p
        self.delta1 = delta1
        self.delta0 = delta0
        self.group_perc = group_perc
        self.sex_prob = sex_prob
        self.pheno_prob = pheno_prob
        self.age_perc = age_perc if age_perc is not None else [1 / 3.0] * 3
        self.region_perc = region_perc if region_perc is not None else [1 / 4.0] * 4
        self.n_nonsensitive = n_nonsensitive
        np.random.seed(seed)

    def sample_X(self, n):
        # Z = six non-sensitive physiological measurements
        Z = np.random.uniform(0, 1, (n, self.n_nonsensitive))
        A_sex = np.random.binomial(1, self.sex_prob, size=n)
        A_pheno = np.random.binomial(1, self.pheno_prob, size=n)
        A_age = np.random.choice(np.arange(len(self.age_perc)), size=n, p=self.age_perc)
        A_region = np.random.choice(np.arange(len(self.region_perc)), size=n, p=self.region_perc)

        X = np.concatenate(
            [
                Z,
                A_sex.reshape(-1, 1),
                A_pheno.reshape(-1, 1),
                A_age.reshape(-1, 1),
                A_region.reshape(-1, 1),
            ],
            axis=1,
        )
        return X

    def compute_prob(self, X):
        P = np.zeros((X.shape[0], self.K))
        u = np.ones(self.K) / self.K  # uniform distribution

        Z1 = X[:, 0]
        Z2 = X[:, 1]
        A_sex = X[:, self.n_nonsensitive + 0]
        A_pheno = X[:, self.n_nonsensitive + 1]
        G_bias = (A_sex == A_pheno).astype(int)

        for i in range(X.shape[0]):
            # Compute latent pi(x) based on Z2
            idx = int(np.floor(Z2[i] * self.K))
            idx = min(max(idx, 0), self.K - 1)
            pi = np.zeros(self.K)
            pi[idx] = 1.0  # deterministic for simplicity

            # Define ambiguous region R_amb
            in_amb = Z1[i] < 0.5  # predefined ambiguous region

            if not in_amb:
                P[i] = pi
            else:
                if G_bias[i] == 1:
                    P[i] = (1 - self.delta1) * pi + self.delta1 * u
                else:
                    P[i] = (1 - self.delta0) * pi + self.delta0 * u
        return P

    def sample_Y(self, X, return_prob=False):
        prob_y = self.compute_prob(X)
        g = np.array([np.random.multinomial(1, prob_y[i]) for i in range(X.shape[0])], dtype=float)
        classes_id = np.arange(self.K)
        y = np.array([np.dot(g[i], classes_id) for i in range(X.shape[0])], dtype=int)
        if return_prob:
            return prob_y, y.astype(int)
        return y.astype(int)


class SyntheticMulticlassDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.sex = torch.tensor(X[:, -4].astype(np.int64), dtype=torch.long)
        self.pheno = torch.tensor(X[:, -3].astype(np.int64), dtype=torch.long)
        self.age = torch.tensor(X[:, -2].astype(np.int64), dtype=torch.long)
        self.region = torch.tensor(X[:, -1].astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.sex[idx],
            self.pheno[idx],
            self.age[idx],
            self.region[idx],
            torch.zeros_like(self.y[idx]),
        )


def build_dataloaders(data_cfg):
    seed = int(data_cfg.get("seed", 42))
    K = int(data_cfg.get("K", 6))
    n_train_cal = int(data_cfg.get("n_use", 100))
    n_test = int(data_cfg.get("test_samples", 20))
    n_nonsensitive = int(data_cfg.get("n_nonsensitive", 6))
    train_size = float(data_cfg.get("train_size", data_cfg.get("train_frac", 0.5)))
    if not (0.0 < train_size < 1.0):
        raise ValueError(f"train_size must be in (0, 1), got {train_size}")
    if n_train_cal < 2:
        raise ValueError(f"n_use must be >= 2 (train+cal), got {n_train_cal}")

    sampler = data_model(
        p=n_nonsensitive + 4,
        delta1=float(data_cfg.get("delta1", 0.9)),
        delta0=float(data_cfg.get("delta0", 0.1)),
        K=K,
        seed=seed,
        n_nonsensitive=n_nonsensitive,
    )

    X_full = sampler.sample_X(n_train_cal)
    y_full = sampler.sample_Y(X_full)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_full, y_full, train_size=train_size, random_state=seed, shuffle=True
    )

    X_test = sampler.sample_X(n_test)
    y_test = sampler.sample_Y(X_test)

    train_ds = SyntheticMulticlassDataset(X_train, y_train)
    cal_ds = SyntheticMulticlassDataset(X_cal, y_cal)
    test_ds = SyntheticMulticlassDataset(X_test, y_test)

    loader_kwargs = {"batch_size": int(data_cfg.get("batch_size", 10))}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    cal_loader = DataLoader(cal_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, cal_loader, test_loader
