from itertools import islice, cycle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple


class SyntheticMulticlassDataset_2(Dataset):
    """
    two-sensitive-attribute DGP.

    Matches the original logic:
      - X ~ Uniform(0,1)^(n,p)
      - overwrite last columns:
          X[:,-1] = groups1
          X[:,-2] = groups2
          X[:,-3] = cyclic noise attr in {1,2,3,4,5}
          X[:,-4] = binned noise attr in {6,7,8,9}
      - label rule:
          if (g1==g2): use delta1
          else: use delta0
          then:
             if X0 < delta and X1 < 0.5: Y uniform on first half classes
             if X0 < delta and X1 >= 0.5: Y uniform on second half classes
             else: deterministic by round(K*X1 - 0.5)

    Returns:
      x, y, group1, group2, attr1, attr2
    where:
      attr1 = X[:,-3]  (cyclic attribute)
      attr2 = X[:,-4]  (binned attribute)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        super().__init__()
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.group1 = torch.tensor(X[:, -1].astype(np.int64), dtype=torch.long)
        self.group2 = torch.tensor(X[:, -2].astype(np.int64), dtype=torch.long)
        self.attr1 = torch.tensor(X[:, -3].astype(np.int64), dtype=torch.long)
        self.attr2 = torch.tensor(X[:, -4].astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.group1[idx],
            self.group2[idx],
            self.attr1[idx],
            self.attr2[idx],
        )


class SyntheticMulticlassDataset_1(Dataset):
    """
    single-sensitive-attribute DGP style.

    Matches the original logic:
      - X ~ Uniform(0,1)^(n,p)
      - overwrite last columns:
          X[:,-1] = group
          X[:,-2] = cyclic noise attr in {1,2,3,4,5}
          X[:,-3] = binned noise attr in {6,7,8,9}
      - label rule:
          if group == 1: use delta1
          else: use delta0
          then:
             if X0 < delta and X1 < 0.5: Y uniform on first half classes
             if X0 < delta and X1 >= 0.5: Y uniform on second half classes
             else: deterministic by round(K*X1 - 0.5)

    Returns:
      x, y, group, attr1, attr2
    where:
      attr1 = X[:,-2]  (cyclic attribute)
      attr2 = X[:,-3]  (binned attribute)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        super().__init__()
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.group = torch.tensor(X[:, -1].astype(np.int64), dtype=torch.long)
        self.attr1 = torch.tensor(X[:, -2].astype(np.int64), dtype=torch.long)
        self.attr2 = torch.tensor(X[:, -3].astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.group[idx],
            self.attr1[idx],
            self.attr2[idx],
        )


class DataModel_2:
    def __init__(
        self,
        p: int,
        delta1: float,
        delta0: float,
        group_perc_1=(0.5, 0.5),
        group_perc_2=(0.5, 0.5),
        K: int = 6,
        seed: int = 2025,
    ):
        assert p >= 4, "Need at least p >= 4 because last 4 columns are overwritten."
        self.p = p
        self.delta1 = delta1
        self.delta0 = delta0
        self.group_perc_1 = list(group_perc_1)
        self.group_perc_2 = list(group_perc_2)
        self.K = K
        self.rng = np.random.default_rng(seed)

    def sample_X(self, n: int) -> np.ndarray:
        X = self.rng.uniform(0.0, 1.0, size=(n, self.p)).astype(np.float32)

        # groups1
        group_alloc_1 = self.rng.multinomial(1, self.group_perc_1, size=n)
        group_label_1 = np.arange(len(self.group_perc_1))
        groups1 = (group_alloc_1 * group_label_1).sum(axis=1).astype(np.int64)
        X[:, -1] = groups1

        # groups2
        group_alloc_2 = self.rng.multinomial(1, self.group_perc_2, size=n)
        group_label_2 = np.arange(len(self.group_perc_2))
        groups2 = (group_alloc_2 * group_label_2).sum(axis=1).astype(np.int64)
        X[:, -2] = groups2

        # noise attr 1: cycle 1,2,3,4,5
        X[:, -3] = np.array(list(islice(cycle([1, 2, 3, 4, 5]), n)), dtype=np.float32)

        # noise attr 2: bins -> 6,7,8,9
        raw = X[:, -4].copy()
        binned = np.digitize(raw, bins=[0.25, 0.5, 0.75], right=False) + 6
        X[:, -4] = binned.astype(np.float32)

        return X

    def compute_prob(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((X.shape[0], self.K), dtype=np.float64)
        K_half = max(self.K // 2, 2)

        for i in range(X.shape[0]):
            g1 = int(X[i, -1])
            g2 = int(X[i, -2])

            if (g1 == 0 and g2 == 0) or (g1 == 1 and g2 == 1):
                delta = self.delta1
            else:
                delta = self.delta0

            if (X[i, 0] < delta) and (X[i, 1] < 0.5):
                P[i, 0:K_half] = 1.0 / K_half
            elif (X[i, 0] < delta) and (X[i, 1] >= 0.5):
                P[i, K_half:self.K] = 1.0 / (self.K - K_half)
            else:
                idx = int(np.round(self.K * X[i, 1] - 0.5))
                idx = max(0, min(idx, self.K - 1))
                P[i, idx] = 1.0
        
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def sample_Y(self, X: np.ndarray, return_prob: bool = False):
        prob_y = self.compute_prob(X)
        g = np.array(
            [self.rng.multinomial(1, prob_y[i]) for i in range(X.shape[0])],
            dtype=np.float32,
        )
        classes_id = np.arange(self.K)
        y = np.array([(g[i] * classes_id).sum() for i in range(X.shape[0])], dtype=np.int64)
        if return_prob:
            return prob_y, y
        return y


class DataModel_1:
    def __init__(
        self,
        p: int,
        delta1: float,
        delta0: float,
        group_perc=(0.5, 0.5),
        K: int = 6,
        seed: int = 2023,
    ):
        assert p >= 3, "Need at least p >= 3 because last 3 columns are overwritten."
        self.p = p
        self.delta1 = delta1
        self.delta0 = delta0
        self.group_perc = list(group_perc)
        self.K = K
        self.rng = np.random.default_rng(seed)

    def sample_X(self, n: int) -> np.ndarray:
        X = self.rng.uniform(0.0, 1.0, size=(n, self.p)).astype(np.float32)

        # group
        group_alloc = self.rng.multinomial(1, self.group_perc, size=n)
        group_label = np.arange(len(self.group_perc))
        groups = (group_alloc * group_label).sum(axis=1).astype(np.int64)
        X[:, -1] = groups

        # noise attr 1: cycle 1,2,3,4,5
        X[:, -2] = np.array(list(islice(cycle([1, 2, 3, 4, 5]), n)), dtype=np.float32)

        # noise attr 2: bins -> 6,7,8,9
        raw = X[:, -3].copy()
        binned = np.digitize(raw, bins=[0.25, 0.5, 0.75], right=False) + 6
        X[:, -3] = binned.astype(np.float32)

        return X

    def compute_prob(self, X: np.ndarray) -> np.ndarray:
        P = np.zeros((X.shape[0], self.K), dtype=np.float64)
        K_half = max(self.K // 2, 2)

        for i in range(X.shape[0]):
            g = int(X[i, -1])
            delta = self.delta1 if g == 1 else self.delta0

            if (X[i, 0] < delta) and (X[i, 1] < 0.5):
                P[i, 0:K_half] = 1.0 / K_half
            elif (X[i, 0] < delta) and (X[i, 1] >= 0.5):
                P[i, K_half:self.K] = 1.0 / (self.K - K_half)
            else:
                idx = int(np.round(self.K * X[i, 1] - 0.5))
                idx = max(0, min(idx, self.K - 1))
                P[i, idx] = 1.0

        P = P / P.sum(axis=1, keepdims=True)
        return P

    def sample_Y(self, X: np.ndarray, return_prob: bool = False):
        prob_y = self.compute_prob(X)
        g = np.array(
            [self.rng.multinomial(1, prob_y[i]) for i in range(X.shape[0])],
            dtype=np.float32,
        )
        classes_id = np.arange(self.K)
        y = np.array([(g[i] * classes_id).sum() for i in range(X.shape[0])], dtype=np.int64)
        if return_prob:
            return prob_y, y
        return y


def build_dataloaders_2(data_cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    workflow:
      1) sample full data of size n_tra_cal
      2) split into train/cal (50/50)
      3) sample test independently
    Expected fields in data_cfg:
      - K, p, delta1, delta0, group1_prob_1, group2_prob_1
      - n_samples   (used as train+cal total)
      - test_samples
      - batch_size
      - seed
    """
    seed = int(getattr(data_cfg, "seed", 42))
    K = int(data_cfg.K)
    p = 10
    n_train_cal = int(data_cfg.n_samples)
    n_test = int(data_cfg.test_samples)

    sampler = DataModel_2(
        p=p,
        delta1=float(data_cfg.delta1),
        delta0=float(data_cfg.delta0),
        group_perc_1=[1 - float(data_cfg.group1_prob_1), float(data_cfg.group1_prob_1)],
        group_perc_2=[1 - float(data_cfg.group2_prob_1), float(data_cfg.group2_prob_1)],
        K=K,
        seed=seed,
    )

    X_full = sampler.sample_X(n_train_cal)
    y_full = sampler.sample_Y(X_full)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_full, y_full, train_size=0.5, random_state=seed
    )

    X_test = sampler.sample_X(n_test)
    y_test = sampler.sample_Y(X_test)

    train_ds = SyntheticMulticlassDataset_2(X_train, y_train)
    cal_ds = SyntheticMulticlassDataset_2(X_cal, y_cal)
    test_ds = SyntheticMulticlassDataset_2(X_test, y_test)

    loader_kwargs = {"batch_size": int(data_cfg.batch_size)}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    cal_loader = DataLoader(cal_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, cal_loader, test_loader


def build_dataloaders_1(data_cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Strict single-sensitive-attribute version.
    Expected fields in data_cfg:
      - K, p, delta1, delta0, group_prob_1
      - n_samples   (used as train+cal total)
      - test_samples
      - batch_size
      - seed
    """
    seed = int(getattr(data_cfg, "seed", 42))
    K = int(data_cfg.K)
    p = 10
    n_train_cal = int(data_cfg.n_samples)
    n_test = int(data_cfg.test_samples)

    sampler = DataModel_1(
        p=p,
        delta1=float(data_cfg.delta1),
        delta0=float(data_cfg.delta0),
        group_perc=[1 - float(data_cfg.group_prob_1), float(data_cfg.group_prob_1)],
        K=K,
        seed=seed,
    )

    X_full = sampler.sample_X(n_train_cal)
    y_full = sampler.sample_Y(X_full)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_full, y_full, train_size=0.5, random_state=seed
    )

    X_test = sampler.sample_X(n_test)
    y_test = sampler.sample_Y(X_test)

    train_ds = SyntheticMulticlassDataset_1(X_train, y_train)
    cal_ds = SyntheticMulticlassDataset_1(X_cal, y_cal)
    test_ds = SyntheticMulticlassDataset_1(X_test, y_test)

    loader_kwargs = {"batch_size": int(data_cfg.batch_size)}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    cal_loader = DataLoader(cal_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, cal_loader, test_loader
