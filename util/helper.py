
import torch
import numpy as np
import pandas as pd
import os

from typing import Any, Dict, List, Optional
import copy
import itertools
import json

from types import SimpleNamespace
from dataset.synthetic import build_dataloaders_1, build_dataloaders_2
from dataset.mimic import build_dataloaders_mimic
from dataset.adult import build_dataloaders_adult
from dataset.nursery import build_dataloaders_nursery
from typing import Dict, Any, Tuple


def build_dataset_and_loaders(
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[Any, Any, Any, str, Any]:
    d_cfg = SimpleNamespace(**data_cfg)
    d_cfg.seed = seed

    if d_cfg.dataset_mode == "mimic":
        tr_loader, ca_loader, te_loader, meta = build_dataloaders_mimic(d_cfg)
        model_cfg["feature_dim"] = meta.feature_dim
        return tr_loader, ca_loader, te_loader, "mimic", meta
    elif d_cfg.dataset_mode == "adult":
        tr_loader, ca_loader, te_loader, meta = build_dataloaders_adult(d_cfg)
        model_cfg["feature_dim"] = meta.feature_dim
        return tr_loader, ca_loader, te_loader, "adult", meta
    elif d_cfg.dataset_mode == "nursery":
        tr_loader, ca_loader, te_loader, meta = build_dataloaders_nursery(d_cfg)
        model_cfg["feature_dim"] = meta.feature_dim
        return tr_loader, ca_loader, te_loader, "nursery", meta
    elif d_cfg.dataset_mode == "bach":
        from dataset.image_data import build_dataloaders_bach
        tr_loader, ca_loader, te_loader, meta = build_dataloaders_bach(d_cfg)
        return tr_loader, ca_loader, te_loader, "bach", meta
    elif d_cfg.dataset_mode == "two_sensitive":
        k = int(model_cfg.get("num_classes", 6))
        syn_cfg = SimpleNamespace(
            K=k,
            delta1=d_cfg.delta1,
            delta0=d_cfg.delta0,
            group1_prob_1=d_cfg.group1_prob_1,
            group2_prob_1=d_cfg.group2_prob_1,
            n_nonsensitive=d_cfg.n_nonsensitive,
            n_samples=d_cfg.n_tra_cal,
            test_samples=d_cfg.test_samples,
            batch_size=d_cfg.batch_size,
            seed=seed,
        )
        tr_loader, ca_loader, te_loader = build_dataloaders_2(syn_cfg)
        model_cfg["feature_dim"] = next(iter(tr_loader))[0].shape[1]
        return tr_loader, ca_loader, te_loader, "two_sensitive", syn_cfg
    elif d_cfg.dataset_mode == "single_sensitive":
        k = int(model_cfg.get("num_classes", 6))
        syn_cfg = SimpleNamespace(
            K=k,
            delta1=d_cfg.delta1,
            delta0=d_cfg.delta0,
            group_prob_1=d_cfg.color_blue_prob,
            n_nonsensitive=d_cfg.n_nonsensitive,
            n_samples=d_cfg.n_tra_cal,
            test_samples=d_cfg.test_samples,
            batch_size=d_cfg.batch_size,
            seed=seed,
        )
        tr_loader, ca_loader, te_loader = build_dataloaders_1(syn_cfg)
        model_cfg["feature_dim"] = next(iter(tr_loader))[0].shape[1]
        return tr_loader, ca_loader, te_loader, "single_sensitive", syn_cfg
    else:
        raise ValueError(f"Unknown dataset_mode: {d_cfg.dataset_mode}")

def pop_seed_sweep_from_grid(grid: Dict[str, List[Any]]):
    for k in ("seeds", "seed", "experiment.seed"):
        if k in grid:
            return grid.pop(k), k
    return None, None


def resolve_seed_list(
    cli_seeds: Optional[List[int]],
    default_seed: int,
    grid_seed_list: Optional[List[int]],
    grid_seed_key: Optional[str],
):
    if cli_seeds is not None and len(cli_seeds) > 0:
        if grid_seed_list is not None:
            print(f'[sweep] Note: ignoring grid "{grid_seed_key}" because --seeds was provided.')
        return cli_seeds
    if grid_seed_list is not None:
        if isinstance(grid_seed_list, (int, np.integer)):
            grid_seed_list = [int(grid_seed_list)]
        elif not isinstance(grid_seed_list, (list, tuple)):
            grid_seed_list = [grid_seed_list]
        if len(grid_seed_list) > 0:
            return [int(s) for s in grid_seed_list]
    return [default_seed]


def clone_cfg(cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(cfg)



# =========================
# Sweep helpers
# =========================

def load_grid_json(grid_json_path: Optional[str]) -> Dict[str, List[Any]]:
    if grid_json_path is None:
        return {}

    with open(grid_json_path, "r", encoding="utf-8") as f:
        grid = json.load(f)

    if not isinstance(grid, dict):
        raise ValueError("grid JSON must be a dict: {param_name: [values...]}")

    normalized = {}
    for k, v in grid.items():
        if not isinstance(v, list):
            raise ValueError(f"Grid entry for {k} must be a list.")
        normalized[k] = v
    return normalized


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values_product = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, vals)) for vals in values_product]


def apply_overrides(
    cfg: Dict[str, Dict[str, Any]],
    override_dict: Dict[str, Any],
):
    for dotted_key, value in override_dict.items():
        if "." not in dotted_key:
            if dotted_key in {"seed", "seeds"}:
                raise ValueError(
                    f'Invalid grid key "{dotted_key}". If you want to sweep over random seeds, '
                    f'use CLI flag "--seeds 0 1 2" or put "seeds": [0,1,2] in grid.json. '
                    f'Other keys must be dotted names like "sgcp_train.lr".'
                )
            raise ValueError(f'Invalid grid key "{dotted_key}". Expected dotted name like "sgcp_train.lr".')

        prefix, attr = dotted_key.split(".", 1)
        if prefix not in cfg:
            raise ValueError(
                f'Unknown config prefix "{prefix}" in grid key "{dotted_key}". '
                f"Allowed: {list(cfg.keys())}"
            )

        if attr not in cfg[prefix]:
            raise ValueError(f'Config "{prefix}" has no field "{attr}" for key "{dotted_key}".')

        cfg[prefix][attr] = value

CONFIG_PREFIXES = (
    "experiment.",
    "dataset.",
    "model.",
    "backbone_train.",
    "proto_train.",
    "sgcp_train.",
)

def aggregate_summary_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across seeds:
      - group by all config columns except experiment.seed
      - compute mean/std/count for numeric metric columns
    """
    if df.empty:
        return df.copy()

    metric_cols = [c for c in df.columns if c.startswith("metric.")]
    numeric_metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_metric_cols:
        return df.copy()

    group_cols = [
        c
        for c in df.columns
        if (
            c == "method"
            or c == "dataset_name"
            or (c.startswith(CONFIG_PREFIXES) and c != "experiment.seed")
        )
    ]

    grouped = df.groupby(group_cols, dropna=False)[numeric_metric_cols]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std(ddof=1).reset_index()
    count_df = grouped.size().reset_index(name="seed_count")

    out = mean_df.copy()
    rename_mean = {c: f"{c}.mean" for c in numeric_metric_cols}
    out = out.rename(columns=rename_mean)

    std_df = std_df.rename(columns={c: f"{c}.std" for c in numeric_metric_cols})
    out = out.merge(std_df, on=group_cols, how="left")
    out = out.merge(count_df, on=group_cols, how="left")

    return out


def print_configs(**configs):
    print("\n=== Config ===")
    for name, cfg in configs.items():
        print(f"[{name}]")
        for k, v in cfg.items():
            print(f"  {k}: {v}")


def flatten_config_dict(prefix: str, cfg) -> dict:
    return {f"{prefix}.{k}": v for k, v in cfg.items()}


def to_serializable(v):
    decimals = get_result_decimals()
    def _conv(x, round_float: bool):
        if isinstance(x, dict):
            return {_conv(k, False): _conv(vv, True) for k, vv in x.items()}
        if isinstance(x, (list, tuple)):
            return [_conv(xx, True) for xx in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            xf = float(x)
            return round(xf, decimals) if round_float else xf
        if isinstance(x, float):
            return round(x, decimals) if round_float else x
        if isinstance(x, np.ndarray):
            return [_conv(xx, True) for xx in x.tolist()]
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                xi = x.item()
                if isinstance(xi, float):
                    return round(xi, decimals) if round_float else xi
                return xi
            return [_conv(xx, True) for xx in x.detach().cpu().tolist()]
        return x
    return _conv(v, False)


def get_result_decimals(default: int = 6) -> int:
    v = os.getenv("RESULT_DECIMALS")
    if v is None:
        return default
    try:
        n = int(v)
    except ValueError:
        return default
    return n if n >= 0 else default


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
