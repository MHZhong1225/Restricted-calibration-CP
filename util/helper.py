import os
from typing import Any

import numpy as np
import torch


def print_configs(**configs):
    print("\n=== Config ===")
    for name, cfg in configs.items():
        print(f"[{name}]")
        for key, value in cfg.items():
            print(f"  {key}: {value}")


def print_run_summary(cfg, device):
    exp = cfg["experiment"]
    data = cfg["dataset"]
    model = cfg["model"]
    backbone = cfg["backbone_train"]
    sgcp = cfg["sgcp_train"]
    print("\n=== Run ===")
    print(
        "  "
        f"dataset={data.get('dataset_mode')} "
        f"seed={exp.get('seed')} "
        f"alpha={exp.get('alpha')} "
        f"device={device}"
    )
    print(
        "  "
        f"n_tra_cal={data.get('n_tra_cal')} "
        f"test_samples={data.get('test_samples')} "
        f"batch_size={data.get('batch_size')}"
    )
    print(
        "  "
        f"classes={model.get('num_classes')} "
        f"prototypes={model.get('num_prototypes')} "
        f"latent_dim={model.get('latent_dim')}"
    )
    print(
        "  "
        f"backbone_epochs={backbone.get('epochs')} "
        f"sgcp_epochs={sgcp.get('epochs')} "
        f"sgcp_lr={sgcp.get('lr')}"
    )


def flatten_config_dict(prefix: str, cfg) -> dict:
    return {f"{prefix}.{key}": value for key, value in cfg.items()}


def get_result_decimals(default: int = 6) -> int:
    try:
        return max(0, int(os.getenv("RESULT_DECIMALS", default)))
    except ValueError:
        return default


def to_serializable(value: Any):
    decimals = get_result_decimals()

    if isinstance(value, dict):
        return {to_serializable(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_serializable(value.tolist())
    if isinstance(value, torch.Tensor):
        return to_serializable(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), decimals)
    if isinstance(value, float):
        return round(value, decimals)
    return value


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
