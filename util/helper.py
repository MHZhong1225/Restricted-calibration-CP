
import torch
import numpy as np
import os

from typing import Any, Dict, List, Optional
import copy
import itertools
import json



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
