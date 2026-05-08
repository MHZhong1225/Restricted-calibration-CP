import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch.nn as nn
import torch

from CP_fairness.sgcp_flow import Backbone
from eval import evaluate_sgcp
from util.helper import flatten_config_dict, print_run_summary, set_seed, to_serializable
from util.train_tool import split_train_loader, train_backbone


def _sync_if_cuda(device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def default_cfg() -> Dict[str, Dict[str, Any]]:
    return {
        "experiment": {
            "seed": 42,
            "alpha": 0.1,
            "cuda": "0",
            "outdir": "results",
            "methods": "sgcp",
        },
        "dataset": {
            "dataset_mode": "syn",
            "n_tra_cal": 0,
            "test_samples": 500,
            "color_blue_prob": 0.10,
            "batch_size": 128,
            "train_frac": 0.5,
            "cal_frac": 0.5,
            "delta1": 0.5,
            "delta0": 0.2,
            "n_nonsensitive": 6,
            "image_backbone": "resnet18",
            "bach_data_dir": "dataset/datasets/bach",
            "mimic_preprocessed_path": "dataset/mimic-iv_processed.csv",
            "mimic_train_frac": 0.6,
            "mimic_cal_frac": 0.2,
            "mimic_label_col": "label",
            "mimic_sensitive_cols": "minority,gender_m,public_insurance",
            "mimic_feature_cols": (
                "age,gender_m,ins_private,ins_medicare,ins_medicaid,"
                "adm_emergency,adm_elective,adm_urgent,marital_married,"
                "marital_single,icu_los,num_diagnoses"
            ),
            "nursery_csv_path": "dataset/nursery/nursery.csv",
        },
        "model": {
            "hidden_dim": 128,
            "latent_dim": 8,
            "num_prototypes": 8,
            "num_classes": 6,
            "temperature": 1.0,
            "stochastic_hidden_dim": 64,
            "stochastic_num_hidden": 2,
            "min_sig": 1e-3,
        },
        "backbone_train": {
            "epochs": 200,
            "lr": 1e-3,
        },
        "sgcp_train": {
            "epochs": 200,
            "lr": 1e-3,
            "lambda_nll": 1e-6,
            "lambda_balance": 1e-2,
            "lambda_rank": 0.5,
            "train_latent_samples": 8,
            "eval_latent_samples": 10,
            "num_score_bins": 20,
            "score_bin_edges": "quantile",
            "hist_smoothing": 1e-3,
            "beta_kl": 1e-3,
            "lambda_score_kl": 1e-3,
            "lambda_score": 1e-3,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SGCP method.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--outdir", type=str, default="results_sgcp_mimic")
    parser.add_argument("--methods", type=str, default="sgcp")
    parser.add_argument("--dataset-mode", type=str, default="syn", choices=["syn", "mimic", "nursery", "bach"])
    parser.add_argument("--grid-json", type=str, default=None)
    parser.add_argument("--n_tra_cal", "--n-tra-cal", "--n-train-cal", dest="n_tra_cal", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument(
        "--image-backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet-b0"],
    )
    parser.add_argument("--bach-data-dir", type=str, default="dataset/datasets/bach")
    parser.add_argument("--mimic-preprocessed-path", type=str, default="dataset/mimic-iv_processed.csv")
    parser.add_argument("--mimic-label-col", type=str, default="label")
    parser.add_argument("--mimic-sensitive-cols", type=str, default="minority,gender_m,public_insurance")
    parser.add_argument(
        "--mimic-feature-cols",
        type=str,
        default=default_cfg()["dataset"]["mimic_feature_cols"],
    )
    parser.add_argument("--nursery-csv-path", type=str, default="dataset/nursery/nursery.csv")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--backbone-epochs", type=int, default=200)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--sgcp-epochs", type=int, default=300)
    parser.add_argument("--sgcp-lr", type=float, default=1e-3)
    parser.add_argument("--sgcp-train-latent-samples", type=int, default=8)
    parser.add_argument("--sgcp-eval-latent-samples", type=int, default=10)
    parser.add_argument("--sgcp-lambda-nll", type=float, default=1e-6)
    parser.add_argument("--sgcp-lambda-balance", type=float, default=1e-2)
    parser.add_argument("--sgcp-lambda-rank", type=float, default=0.5)
    return parser


def config_from_args(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    cfg = default_cfg()
    cfg["experiment"].update(
        {
            "seed": args.seed,
            "alpha": args.alpha,
            "cuda": args.cuda,
            "outdir": args.outdir + f"_{args.alpha}",
            "methods": args.methods,
        }
    )
    cfg["dataset"].update(
        {
            "dataset_mode": args.dataset_mode,
            "n_tra_cal": args.n_tra_cal,
            "test_samples": args.test_samples,
            "batch_size": args.batch_size,
            "train_frac": args.train_frac,
            "image_backbone": args.image_backbone,
            "bach_data_dir": args.bach_data_dir,
            "mimic_preprocessed_path": args.mimic_preprocessed_path,
            "mimic_label_col": args.mimic_label_col,
            "mimic_sensitive_cols": args.mimic_sensitive_cols,
            "mimic_feature_cols": args.mimic_feature_cols,
            "nursery_csv_path": args.nursery_csv_path,
        }
    )
    cfg["model"].update(
        {
            "num_classes": args.num_classes,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_prototypes": args.num_prototypes,
            "temperature": args.temperature,
        }
    )
    cfg["backbone_train"].update({"epochs": args.backbone_epochs, "lr": args.backbone_lr})
    cfg["sgcp_train"].update(
        {
            "epochs": args.sgcp_epochs,
            "lr": args.sgcp_lr,
            "train_latent_samples": args.sgcp_train_latent_samples,
            "eval_latent_samples": args.sgcp_eval_latent_samples,
            "lambda_nll": args.sgcp_lambda_nll,
            "lambda_balance": args.sgcp_lambda_balance,
            "lambda_rank": args.sgcp_lambda_rank,
        }
    )
    return cfg


def build_dataset_and_loaders(cfg: Dict[str, Dict[str, Any]]):
    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    common = {
        "seed": cfg["experiment"]["seed"],
        "K": model_cfg["num_classes"],
            "n_use": data_cfg["n_tra_cal"],
        "test_samples": data_cfg["test_samples"],
        "batch_size": data_cfg["batch_size"],
        "train_frac": data_cfg["train_frac"],
    }
    mode = data_cfg["dataset_mode"]
    if mode == "syn":
        from dataset.syn import build_dataloaders

        return build_dataloaders(
            {
                **common,
                "n_use": data_cfg["n_tra_cal"] + 500,
                "delta1": data_cfg["delta1"],
                "delta0": data_cfg["delta0"],
                "group_prob_1": data_cfg["color_blue_prob"],
                "n_nonsensitive": data_cfg["n_nonsensitive"],
            }
        )
    if mode == "mimic":
        from dataset.tabular import build_dataloaders_mimic

        return build_dataloaders_mimic({**common, **data_cfg})
    if mode == "nursery":
        from dataset.tabular import build_dataloaders_nursery

        return build_dataloaders_nursery({**common, **data_cfg, "n_use": data_cfg["n_tra_cal"] + 500})
    if mode == "bach":
        from dataset.image_data import build_dataloaders_bach

        return build_dataloaders_bach(
            {**common, **data_cfg, "n_use": data_cfg["n_tra_cal"], "test_samples": 100}
        )
    raise ValueError(f"Unknown dataset_mode={mode!r}")


def create_image_backbone(backbone_name: str, num_classes: int, pretrained: bool = True):
    from torchvision import models

    if backbone_name == "resnet18":
        print("Loading pretrained ResNet-18 backbone.")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif backbone_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif backbone_name == "efficientnet-b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()

        class EfficientNetBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.extractor = model
                self.classifier = nn.Linear(feature_dim, num_classes)
                self.feature_dim = feature_dim

            def forward(self, x):
                h = self.extractor(x)
                return self.classifier(h), h

        return EfficientNetBackbone()
    else:
        raise ValueError(f"Unsupported image_backbone={backbone_name!r}")

    feature_dim = model.fc.in_features
    model.fc = nn.Identity()

    class ImageBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = model
            self.classifier = nn.Linear(feature_dim, num_classes)
            self.feature_dim = feature_dim

        def forward(self, x):
            h = self.extractor(x)
            return self.classifier(h), h

    return ImageBackbone()


def get_cached_backbone_path(cfg: Dict[str, Dict[str, Any]]) -> str:
    outdir = os.path.join(cfg["experiment"]["outdir"], "checkpoints")
    os.makedirs(outdir, exist_ok=True)
    dataset_mode = cfg["dataset"]["dataset_mode"]
    bb_name = cfg["dataset"].get("image_backbone", "resnet18") if dataset_mode == "bach" else "mlp"
    task_suffix = f"_{cfg['dataset']['mimic_label_col']}" if dataset_mode == "mimic" else ""
    return os.path.join(
        outdir,
        (
            f"{dataset_mode}_{bb_name}{task_suffix}_n{cfg['dataset'].get('n_tra_cal', 0)}"
            f"_lr{cfg['backbone_train']['lr']}_bs{cfg['dataset']['batch_size']}"
            f"_npro{cfg['model'].get('num_prototypes', 0)}_{cfg['dataset']['train_frac']}.pth"
        ),
    )


def load_grid_json(path: Optional[str]) -> Dict[str, list]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    if not isinstance(grid, dict):
        raise ValueError("grid JSON must be an object.")
    return {k: (v if isinstance(v, list) else [v]) for k, v in grid.items()}


def apply_overrides(cfg: Dict[str, Dict[str, Any]], overrides: Dict[str, Any]) -> None:
    aliases = {"dataset.n_train_cal": "dataset.n_tra_cal"}
    for key, value in overrides.items():
        key = aliases.get(key, key)
        if key in {"seed", "seeds"}:
            continue
        if "." not in key:
            raise ValueError(f'Grid key "{key}" must be dotted, e.g. "sgcp_train.lr".')
        section, name = key.split(".", 1)
        if section not in cfg or name not in cfg[section]:
            raise ValueError(f'Unknown config key "{key}".')
        cfg[section][name] = value


def expand_grid(grid: Dict[str, list]):
    if not grid:
        yield {}
        return
    keys = [k for k in grid.keys() if k not in {"seed", "seeds"}]
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def save_results(metrics: Dict[str, Any], cfg: Dict[str, Dict[str, Any]], outdir: str) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    row: Dict[str, Any] = {}
    for section in ("experiment", "dataset", "model", "backbone_train", "sgcp_train"):
        row.update(flatten_config_dict(section, cfg[section]))
    if set(metrics.keys()) == {"SGCP"} and isinstance(metrics["SGCP"], dict):
        metrics = metrics["SGCP"]

    for key, value in metrics.items():
        row[f"metric.{key}"] = to_serializable(value)

    csv_path = os.path.join(outdir, "summary.csv")
    existing_rows = []
    fieldnames = list(row.keys())
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
    return csv_path


def infer_num_classes(*loaders) -> int:
    max_label = -1
    for loader in loaders:
        for batch in loader:
            max_label = max(max_label, int(batch[1].max().item()))
    if max_label < 0:
        raise ValueError("Could not infer number of classes from empty loaders.")
    return max_label + 1


def run_experiment(cfg: Dict[str, Dict[str, Any]], outdir: Optional[str] = None):
    exp_cfg = SimpleNamespace(**cfg["experiment"], dataset_mode=cfg["dataset"]["dataset_mode"])
    set_seed(exp_cfg.seed)
    device = torch.device(f"cuda:{exp_cfg.cuda}" if torch.cuda.is_available() else "cpu")

    train_loader, cal_loader, test_loader = build_dataset_and_loaders(cfg)
    if cfg["dataset"]["dataset_mode"] != "syn":
        cfg["model"]["num_classes"] = infer_num_classes(train_loader, cal_loader, test_loader)
    model_cfg = SimpleNamespace(**cfg["model"])
    sgcp_cfg = SimpleNamespace(**cfg["sgcp_train"])

    print_run_summary(cfg, device)

    train_model_loader, train_cal_loader = split_train_loader(train_loader, seed=exp_cfg.seed)

    if cfg["dataset"]["dataset_mode"] == "bach":
        backbone = create_image_backbone(cfg["dataset"]["image_backbone"], model_cfg.num_classes)
    else:
        input_dim = next(iter(train_loader))[0].shape[1]
        backbone = Backbone(
            input_dim=input_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_classes=model_cfg.num_classes,
        )
    cached_backbone_path = get_cached_backbone_path(cfg)
    if os.path.exists(cached_backbone_path):
        print(f"[runtime] Loading cached backbone from {cached_backbone_path}")
        backbone.to(device)
        backbone.load_state_dict(torch.load(cached_backbone_path, map_location=device))
    else:
        print(f"[runtime] Training backbone and caching to {cached_backbone_path}")
        _sync_if_cuda(device)
        backbone = train_backbone(
            backbone,
            train_loader=train_model_loader,
            epochs=cfg["backbone_train"]["epochs"],
            lr=cfg["backbone_train"]["lr"],
            device=device,
        )
        torch.save(backbone.state_dict(), cached_backbone_path)

    metrics, _assign_model, _sgcp_obj = evaluate_sgcp(
        backbone=backbone,
        train_loader=train_model_loader,
        train_cal_loader=train_cal_loader,
        cal_loader=cal_loader,
        test_loader=test_loader,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        sgcp_cfg=sgcp_cfg,
        device=device,
    )

    print("\n=== SGCP ===")
    metrics_to_print = metrics.get("SGCP", metrics)
    for key, value in metrics_to_print.items():
        print(f"{key}: {value}")

    result_dir = outdir or cfg["experiment"]["outdir"]
    summary_path = save_results(metrics, cfg, result_dir)
    print(f"\nSaved summary: {summary_path}")
    return {"metrics": metrics, "summary_path": summary_path}
