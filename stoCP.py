import argparse
import sys
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional
import pandas as pd
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm

from pathlib import Path
from dataset.synthetic import build_dataloaders_1, build_dataloaders_2
from dataset.mimic import build_dataloaders_mimic
from dataset.adult import build_dataloaders_adult
from dataset.nursery import build_dataloaders_nursery
from eval import evaluate_sgcp
from util.helper import *
from util.train_tool import train_backbone
from SelectiveCI_fairness.sgcp_flow import Backbone


def _create_backbone(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    from torchvision import models
    if backbone_name == 'resnet18':
        print("Loading pretrained ResNet-18 backbone.")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'resnet50':
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
        
    return backbone, num_ftrs

def _round_metric_columns(df: pd.DataFrame, decimals: Optional[int] = None) -> pd.DataFrame:
    if decimals is None:
        decimals = get_result_decimals()
    for c in [col for col in df.columns if col.startswith("metric.")]:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(decimals)
    return df

def default_cfg() -> Dict[str, Dict[str, Any]]:
    return {
        "experiment": {
            "seed": 42, "alpha": 0.1, "hard_cluster_seed": 42,
            "outdir": "results", "methods": "sgcp", "make_intro_figure": False,
        },
        "dataset": {
            "n_tra_cal": 0, "test_samples": 500, "color_blue_prob": 0.10,
            "group1_prob_1": 0.5, "group2_prob_1": 0.5, "delta1": 0.5, "delta0": 0.2,
            "n_nonsensitive": 6, "batch_size": 128, "dataset_mode": "single_sensitive",
            "mimic_preprocessed_path": "dataset/mimic_iv_processed.csv", "mimic_train_frac": 0.6, "mimic_cal_frac": 0.2,
            "mimic_label_col": "label", "mimic_sensitive_col": "minority", "mimic_age_col": "age",
            "mimic_region_col": "", "mimic_feature_cols": "", "mimic_id_cols": "SUBJECT_ID,HADM_ID",
        },
        "model": {
            "hidden_dim": 128, "feature_dim": 32, "latent_dim": 8, "num_prototypes": 8,
            "num_classes": 6, "temperature": 1.0, "stochastic_hidden_dim": 64,
            "stochastic_num_hidden": 2, "min_sig": 1e-3,
        },
        "backbone_train": {"epochs": 200, "lr": 1e-3},
        "sgcp_train": {
            "epochs": 200, "lr": 1e-3, "beta_kl": 1e-6, "lambda_balance": 1e-2,
            "lambda_score": 0.5, "train_latent_samples": 8, "eval_latent_samples": 10,
            "num_score_bins": 20, "score_bin_edges": "quantile", "hist_smoothing": 1e-3,
        },
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default='sgcp')
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--outdir", type=str, default="results_sgcp_mimic")
    
    parser.add_argument("--dataset-mode", type=str, default=None,
                        choices=["single_sensitive", "two_sensitive", "mimic", "adult", "nursery", "bach"])
    
    parser.add_argument("--mimic-label-col", type=str, default="label", 
                        help="Task label: 'label' (Mortality) or 'icu_delay_class' (Resource delay)")
    parser.add_argument("--mimic-feature-cols", type=str, default="age,gender_m,ins_private,ins_medicare,ins_medicaid,adm_emergency,adm_elective,adm_urgent,marital_married,marital_single,icu_los,num_diagnoses", 
                        help="Comma-separated list of feature columns for MIMIC")
    parser.add_argument("--mimic-sensitive-cols", type=str, default="minority,gender_m,public_insurance",
                        help="Comma-separated sensitive attributes (e.g., 'minority,gender_m,public_insurance')")


    parser.add_argument("--n_tra_cal", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--make-intro-figure", action="store_true")

    parser.add_argument("--image-backbone", type=str, default="resnet18", 
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=6)
    
    parser.add_argument("--backbone-epochs", type=int, default=200)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    
    parser.add_argument("--soft-epochs", type=int, default=100)
    parser.add_argument("--soft-lr", type=float, default=1e-2)
    
    parser.add_argument("--sgcp-epochs", type=int, default=300)
    parser.add_argument("--sgcp-lr", type=float, default=1e-3)

    parser.add_argument("--grid-json", type=str, default=None)
    parser.add_argument("--sweep-name", type=str, default="default_sweep")

    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="SGCP")
    parser.add_argument("--wandb-name", type=str, default=None)

    return parser

def config_from_args(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    cfg = default_cfg()
    cfg["experiment"].update({
            "alpha": args.alpha, 
            "outdir": args.outdir + f"_{args.alpha}", 
            "methods": args.methods,
            "seed": args.seed, 
            "cuda": args.cuda, 
            "use_wandb": getattr(args, "use_wandb", False),
            "wandb_project": getattr(args, "wandb_project", "SGCP") + f"_{args.alpha}",
        })
    
    # 通用数据集参数
    cfg["dataset"].update({
            "n_tra_cal": args.n_tra_cal, 
            "test_samples": args.test_samples, 
            "batch_size": args.batch_size,
            "dataset_mode": args.dataset_mode, 
            "image_backbone": getattr(args, "image_backbone", "resnet18"),
        })
    cfg["model"].update({
        "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
        "num_prototypes": args.num_prototypes, "num_classes": args.num_classes,
    })
    # 
    if args.dataset_mode == "mimic":
        cfg["dataset"].update({
            "mimic_label_col": args.mimic_label_col,
            "mimic_sensitive_col": args.mimic_sensitive_col,
            "mimic_feature_cols": args.mimic_feature_cols if args.mimic_label_col == "label" else "age,gender_m,ins_private,ins_medicare,ins_medicaid,adm_emergency,adm_elective,adm_urgent,marital_married,marital_single,num_diagnoses",
        })

        cfg["model"].update({
            "num_classes": 2 if args.mimic_label_col == "label" else 3,
        })

    cfg["backbone_train"].update({"epochs": args.backbone_epochs, "lr": args.backbone_lr})
    cfg["sgcp_train"].update({"epochs": args.sgcp_epochs, "lr": args.sgcp_lr})
    return cfg

def save_run_results(
    results,
    cfg: Dict[str, Dict[str, Any]],
    outdir="results",
):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    def _reorder_covgap_columns(df: pd.DataFrame) -> pd.DataFrame:
        anchor = "metric.attr2_set_sizes"
        preferred = [
            "metric.covgap",
            "metric.covgap_attr1",
            "metric.covgap_attr2",
            "metric.covgap_attr3",
        ]
        covgap_cols = [c for c in preferred if c in df.columns] + [
            c for c in df.columns if c.startswith("metric.covgap") and c not in preferred
        ]
        if len(covgap_cols) == 0:
            return df
        base_cols = [c for c in df.columns if c not in covgap_cols]
        if anchor in base_cols:
            idx = base_cols.index(anchor) + 1
            cols = base_cols[:idx] + covgap_cols + base_cols[idx:]
        else:
            cols = base_cols + covgap_cols
        return df.loc[:, cols]

    base_info = {}
    
    exp_cfg = cfg["experiment"]
    base_info["experiment.seed"] = exp_cfg.get("seed")
    base_info["experiment.alpha"] = exp_cfg.get("alpha")
    
    ds_cfg = cfg["dataset"]
    dataset_mode = ds_cfg.get("dataset_mode")
    base_info["dataset.dataset_mode"] = dataset_mode
    base_info["dataset.batch_size"] = ds_cfg.get("batch_size")
    if dataset_mode in ["single_sensitive", "two_sensitive", "mimic", "adult", "nursery"]:
        base_info["dataset.n_tra_cal"] = ds_cfg.get("n_tra_cal")
        base_info["dataset.test_samples"] = ds_cfg.get("test_samples")
    
    if dataset_mode == "mimic":
        base_info["dataset.mimic_label"] = ds_cfg.get("mimic_label_col")
        base_info["dataset.mimic_sensitive"] = ds_cfg.get("mimic_sensitive_col")
        
    if dataset_mode == "single_sensitive":
        base_info["dataset.color_blue_prob"] = ds_cfg.get("color_blue_prob")
    if dataset_mode == "two_sensitive":
        base_info["dataset.group1_prob_1"] = ds_cfg.get("group1_prob_1")
        base_info["dataset.group2_prob_1"] = ds_cfg.get("group2_prob_1")
    if dataset_mode == "bach":
        base_info["dataset.image_backbone"] = ds_cfg.get("image_backbone")

    model_cfg = cfg["model"]
    base_info["model.num_classes"] = model_cfg.get("num_classes")
    base_info["model.num_prototypes"] = model_cfg.get("num_prototypes")
    base_info["model.latent_dim"] = model_cfg.get("latent_dim")

    base_info.update(flatten_config_dict("backbone_train", cfg["backbone_train"]))
    base_info.update(flatten_config_dict("sgcp_train", cfg["sgcp_train"]))

    rows = []
    for method_name, metrics in results.items():
        row = dict(base_info)
        row["method"] = method_name

        if isinstance(metrics, dict):
            for k, v in metrics.items():
                row[f"metric.{k}"] = to_serializable(v)
        else:
            row["metric.value"] = to_serializable(metrics)

        rows.append(row)

    run_df = pd.DataFrame(rows)
    run_df = _reorder_covgap_columns(run_df)
    run_df = _round_metric_columns(run_df)

    master_csv_path = os.path.join(outdir, "all_runs_summary.csv")
    if os.path.exists(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
        merged = pd.concat([master_df, run_df], ignore_index=True)
        merged = _reorder_covgap_columns(merged)
        merged = _round_metric_columns(merged)
        merged.to_csv(master_csv_path, index=False)
    else:
        run_df.to_csv(master_csv_path, index=False)

    print(f"Updated summary   : {master_csv_path}\n")
    return run_df, master_csv_path

def run_name_for_cfg(cfg: Dict[str, Dict[str, Any]]) -> str:
    dataset_name = cfg['dataset'].get('dataset_mode', 'unknown')
    seed = cfg['experiment']['seed']
    proto = cfg['model']['num_prototypes']
    lr = cfg['sgcp_train']['lr']
    return f"{dataset_name}_s{seed}_proto{proto}_lr{lr}"

def get_cached_backbone_path(cfg: Dict[str, Dict[str, Any]]) -> str:
    outdir = os.path.join(cfg["experiment"]["outdir"], "checkpoints")
    os.makedirs(outdir, exist_ok=True) 
    
    dataset_mode = cfg["dataset"]["dataset_mode"]
    if dataset_mode == "bach":
        bb_name = cfg["dataset"].get("image_backbone", "resnet18")
    else:
        bb_name = "mlp"
        
    lr = cfg["backbone_train"]["lr"]
    batch_size = cfg["dataset"]["batch_size"]
    
    # 根据不同任务缓存不同权重
    task_suffix = ""
    if dataset_mode == "mimic":
        task_suffix = f"_{cfg['dataset']['mimic_label_col']}"
        
    return os.path.join(outdir, f"{dataset_mode}_{bb_name}{task_suffix}_n{cfg['dataset'].get('n_tra_cal', 0)}_lr{lr}_bs{batch_size}.pth")

def build_dataset_and_loaders(data_cfg: Dict[str, Any], model_cfg: Dict[str, Any], seed: int):
    if data_cfg["dataset_mode"] == "mimic":
        mimic_cfg = SimpleNamespace(
            mimic_preprocessed_path=data_cfg.get("mimic_preprocessed_path", "dataset/mimic_iv_processed.csv"),
            n_use=data_cfg.get("n_tra_cal", 0),
            train_frac=data_cfg.get("mimic_train_frac", 0.6),
            cal_frac=data_cfg.get("mimic_cal_frac", 0.2),
            label_col=data_cfg.get("mimic_label_col", "label"),
            sensitive_cols=data_cfg.get("mimic_sensitive_col", "minority"),
            age_col=data_cfg.get("mimic_age_col", "age"),
            region_col=(data_cfg.get("mimic_region_col") or None),
            feature_cols=(data_cfg.get("mimic_feature_cols") or None),
            id_cols=(data_cfg.get("mimic_id_cols") or None),
            batch_size=data_cfg["batch_size"],
            seed=seed,
        )
        train_loader, cal_loader, test_loader, meta = build_dataloaders_mimic(mimic_cfg)
        return train_loader, cal_loader, test_loader, meta

    if data_cfg["dataset_mode"] == "adult":
        adult_cfg = SimpleNamespace(
            adult_csv_path="dataset/adult.csv",
            n_use=data_cfg.get("n_tra_cal", 0),
            train_frac=data_cfg.get("mimic_train_frac", 0.6),
            cal_frac=data_cfg.get("mimic_cal_frac", 0.2),
            batch_size=data_cfg["batch_size"],
            seed=seed,
        )
        train_loader, cal_loader, test_loader, meta = build_dataloaders_adult(adult_cfg)
        return train_loader, cal_loader, test_loader, meta

    if data_cfg["dataset_mode"] == "nursery":
        nursery_cfg = SimpleNamespace(
            nursery_csv_path="dataset/nursery/nursery.csv",
            n_use=data_cfg.get("n_tra_cal", 0),
            train_frac=data_cfg.get("mimic_train_frac", 0.6),
            cal_frac=data_cfg.get("mimic_cal_frac", 0.2),
            batch_size=data_cfg["batch_size"],
            seed=seed,
        )
        train_loader, cal_loader, test_loader, meta = build_dataloaders_nursery(nursery_cfg)
        return train_loader, cal_loader, test_loader, meta

    if data_cfg["dataset_mode"] == "bach":
        from dataset.image_data import build_dataloaders_bach
        bach_cfg = SimpleNamespace(
            image_data_dir="/home/ubuntu/zmh/BrCPT/datasets/bach",
            batch_size=data_cfg["batch_size"],
            seed=seed,
        )
        train_loader, cal_loader, test_loader, meta = build_dataloaders_bach(bach_cfg)
        return train_loader, cal_loader, test_loader, meta

    if data_cfg["dataset_mode"] == "two_sensitive":
        syn_cfg = SimpleNamespace(
            K=model_cfg["num_classes"],
            delta1=data_cfg["delta1"],
            delta0=data_cfg["delta0"],
            group1_prob_1=data_cfg["group1_prob_1"],
            group2_prob_1=data_cfg["group2_prob_1"],
            n_nonsensitive=data_cfg["n_nonsensitive"],
            n_samples=data_cfg["n_tra_cal"],
            test_samples=data_cfg["test_samples"],
            batch_size=data_cfg["batch_size"],
            seed=seed,
        )
        train_loader, cal_loader, test_loader = build_dataloaders_2(syn_cfg)
        return train_loader, cal_loader, test_loader, syn_cfg

    else:
        syn_cfg = SimpleNamespace(
            K=model_cfg["num_classes"],
            delta1=data_cfg["delta1"],
            delta0=data_cfg["delta0"],
            group_prob_1=data_cfg["color_blue_prob"],
            n_nonsensitive=data_cfg["n_nonsensitive"],
            n_samples=data_cfg["n_tra_cal"],
            test_samples=data_cfg["test_samples"],
            batch_size=data_cfg["batch_size"],
            seed=seed,
        )
        train_loader, cal_loader, test_loader = build_dataloaders_1(syn_cfg)
        return train_loader, cal_loader, test_loader, syn_cfg

# =========================
# Core run
# =========================

def run_experiment(
    cfg: Dict[str, Dict[str, Any]],
    exp_cfg: SimpleNamespace,
):
    model_cfg = SimpleNamespace(**cfg["model"])
    sgcp_cfg = SimpleNamespace(**cfg["sgcp_train"])

    set_seed(exp_cfg.seed)
    device = torch.device(f"cuda:{exp_cfg.cuda}" if torch.cuda.is_available() else "cpu")

    print_configs(
        experiment=cfg["experiment"],
        dataset=cfg["dataset"],
        model=cfg["model"],
        backbone_train=cfg["backbone_train"],
        sgcp_train=cfg["sgcp_train"],
    )
    print(f"[runtime] device: {device}\n")

    if getattr(exp_cfg, "use_wandb", False):
        wandb.init(
            project=exp_cfg.wandb_project,
            name=run_name_for_cfg(cfg), 
            config=cfg,                 
            reinit=True                 
        )
        
    train_loader, cal_loader, test_loader, syn_cfg = build_dataset_and_loaders(
        data_cfg=cfg["dataset"],
        model_cfg=cfg["model"],
        seed=exp_cfg.seed,
    )

    print(f"[runtime] dataset_mode: {cfg['dataset']['dataset_mode']}")
    print(f"[runtime] dataset_cfg: {vars(syn_cfg)}\n")

    if hasattr(syn_cfg, "num_classes"):
        model_cfg.num_classes = syn_cfg.num_classes
        cfg["model"]["num_classes"] = syn_cfg.num_classes
        
    is_image = getattr(syn_cfg, "is_image", False)

    if is_image:
        image_backbone_name = cfg["dataset"].get("image_backbone", "resnet18")
        feature_extractor, feature_dim = _create_backbone(image_backbone_name)
        cfg["model"]["feature_dim"] = feature_dim
        
        class ImageBackbone(nn.Module):
            def __init__(self, extractor, feat_dim, num_classes):
                super().__init__()
                self.extractor = extractor
                self.classifier = nn.Linear(feat_dim, num_classes)
                self.feature_dim = feat_dim
            def forward(self, x):
                feats = self.extractor(x)
                logits = self.classifier(feats)
                return logits, feats
                
        backbone = ImageBackbone(feature_extractor, feature_dim, model_cfg.num_classes)
    else:
        backbone = Backbone(
            input_dim=next(iter(train_loader))[0].shape[1],
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
        backbone = train_backbone(
            backbone,
            train_loader,
            epochs=cfg["backbone_train"]["epochs"],
            lr=cfg["backbone_train"]["lr"],
            device=device,
        )
        torch.save(backbone.state_dict(), cached_backbone_path)


    metrics, sg_assign, _sg_cp = evaluate_sgcp(
        backbone=backbone,
        train_loader=train_loader,
        cal_loader=cal_loader,
        test_loader=test_loader,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        sgcp_cfg=sgcp_cfg,
        device=device,
    )

    if cfg["dataset"]["dataset_mode"] == "mimic" and cfg["experiment"].get("methods") == "sgcp" and cfg["experiment"].get("make_intro_figure", False):
        from figures.intro_borrowing_mimic.make_intro_borrowing_mimic_figure import make_intro_borrowing_mimic_figure

        out_dir = os.path.join(exp_cfg.outdir, "figures_intro")
        make_intro_borrowing_mimic_figure(
            backbone=backbone,
            assign_model=sg_assign,
            cal_loader=cal_loader,
            test_loader=test_loader,
            out_dir=out_dir,
            device=str(device),
            n_latent_samples=int(getattr(sgcp_cfg, "eval_latent_samples", 10)),
        )

    for title, metrics0 in metrics.items():
            print(f"\n=== {title} ===")
            print(metrics0)
            
            if getattr(exp_cfg, "use_wandb", False):
                wandb_log_dict = {f"{title}/{k}": v for k, v in metrics0.items() if isinstance(v, (int, float))}
                wandb.log(wandb_log_dict)

    if getattr(exp_cfg, "use_wandb", False):
            wandb.finish()

    run_df, master_csv_path = save_run_results(
        results=metrics,
        cfg=cfg,
        outdir=exp_cfg.outdir,
    )

    return {
        "metrics": metrics,
        "run_df": run_df,
        "master_csv_path": master_csv_path,
    }
