import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import DataLoader, Dataset

class SyntheticMulticlassDataset(Dataset):
    """
    Features:
      - Color: Blue / Grey (sensitive)
      - AgeGroup: 5 levels (sensitive, but not actually harmful in the DGP)
      - Region: 4 levels (sensitive, but not actually harmful in the DGP)
      - 6 non-sensitive U[0,1] covariates

    Label rule follows Appendix A7.1.1:
      - if Color == Blue and X1 < 0.5:
            Y ~ Uniform({0,1,2})
      - if Color == Blue and X1 >= 0.5:
            Y ~ Uniform({3,4,5})
      - if Color == Grey:
            Y is deterministic from which sixth of [0,1] contains X1

    We return:
      x, y, color, age, region
    """

    def __init__(
        self,
        n_samples: int,
        color_blue_prob: float = 0.10,
        n_nonsensitive: int = 6,
        split: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        assert n_nonsensitive >= 1

        rng = np.random.default_rng(seed)
        age_cycle = np.array([0, 1, 2, 3, 4], dtype=np.int64)

        xs, ys = [], []
        colors, ages, regions = [], [], []

        for i in range(n_samples):
            color = int(rng.random() < color_blue_prob)   # 1 = Blue, 0 = Grey
            age = int(age_cycle[i % 5])                   # cyclically repeated
            region = int(rng.integers(0, 4))              # uniform among 4 regions

            nonsens = rng.uniform(0.0, 1.0, size=n_nonsensitive).astype(np.float32)
            x1 = float(nonsens[0])

            if color == 1:
                y = int(rng.integers(0, 3)) if x1 < 0.5 else int(rng.integers(3, 6))
            else:
                y = min(int(np.floor(6.0 * x1)), 5)

            x = np.concatenate(
                [
                    np.array(
                        [
                            float(color),
                            age / 4.0,
                            region / 3.0,
                        ],
                        dtype=np.float32,
                    ),
                    nonsens.astype(np.float32),
                ],
                axis=0,
            )

            xs.append(x)
            ys.append(y)
            colors.append(color)
            ages.append(age)
            regions.append(region)

        self.x = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.long)
        self.color = torch.tensor(np.array(colors), dtype=torch.long)
        self.age = torch.tensor(np.array(ages), dtype=torch.long)
        self.region = torch.tensor(np.array(regions), dtype=torch.long)
        self.split = split

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.color[idx], self.age[idx], self.region[idx]



def build_dataloaders(data_cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common_dataset_kwargs = {
        "color_blue_prob": data_cfg.color_blue_prob,
        "n_nonsensitive": data_cfg.n_nonsensitive,
    }
    train_ds = SyntheticMulticlassDataset(
        n_samples=data_cfg.train_samples,
        split="train",
        seed=data_cfg.train_seed,
        **common_dataset_kwargs,
    )
    cal_ds = SyntheticMulticlassDataset(
        n_samples=data_cfg.cal_samples,
        split="cal",
        seed=data_cfg.cal_seed,
        **common_dataset_kwargs,
    )
    test_ds = SyntheticMulticlassDataset(
        n_samples=data_cfg.test_samples,
        split="test",
        seed=data_cfg.test_seed,
        **common_dataset_kwargs,
    )

    loader_kwargs = {"batch_size": data_cfg.batch_size}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    cal_loader = DataLoader(cal_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, cal_loader, test_loader



