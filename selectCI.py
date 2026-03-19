import argparse

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from tqdm.auto import tqdm

import pandas as pd
import torch

from data.synthetic import build_dataloaders_1, build_dataloaders_2
from eval import evaluate_sgcp
from util.helper import *

from util.train_tool import train_backbone

from SelectiveCI_fairness.sgcp_flow import Backbone


def _round_metric_columns(df: pd.DataFrame, decimals: Optional[int] = None) -> pd.DataFrame:
    if decimals is None:
        decimals = get_result_decimals()
    metric_cols = [c for c in df.columns if c.startswith("metric.")]
    for c in metric_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].round(decimals)
    return df


# =========================
# Config
# =========================

def default_cfg() -> Dict[str, Dict[str, Any]]:
    return {
        "experiment": {
            "seed": 42,
            "alpha": 0.1,
            "hard_cluster_seed": 42,
            "outdir": "results",
            "methods": "sgcp",
        },
        "dataset": {
            "n_tra_cal": int,
            "test_samples": 500,
            "color_blue_prob": 0.10,
            "group1_prob_1": 0.5,
            "group2_prob_1": 0.5,
            "delta1": 0.5,
            "delta0": 0.2,
            "n_nonsensitive": 6,
            "batch_size": 128,
            "dataset_mode": "single_sensitive",
        },
        "model": {
            "hidden_dim": 128,
            "feature_dim": 32,
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
            "beta_kl": 1e-6,
            "lambda_balance": 1e-2,
            "lambda_score": 0.5,
            "train_latent_samples": 8,
            "eval_latent_samples": 10,
            "num_score_bins": 20,
            "score_bin_edges": "quantile",
            "hist_smoothing": 1e-3,
        },
    }


# =========================
# Argparse
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods",type=str, default='sgcp')
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional multi-seed sweep, e.g. --seeds 0 1 2 3")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--outdir", type=str, default="results_sgcp")

    # dataset
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default="single_sensitive",
        choices=["single_sensitive", "two_sensitive"],
    )
    parser.add_argument("--n_tra_cal", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--color-blue-prob", type=float, default=0.10)
    parser.add_argument("--group1-prob-1", type=float, default=0.5)
    parser.add_argument("--group2-prob-1", type=float, default=0.5)
    parser.add_argument("--delta1", type=float, default=0.5)
    parser.add_argument("--delta0", type=float, default=0.2)
    parser.add_argument("--n-nonsensitive", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)

    # model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stochastic-hidden-dim", type=int, default=64)
    parser.add_argument("--stochastic-num-hidden", type=int, default=2)
    parser.add_argument("--min-sig", type=float, default=1e-3)

    # backbone
    parser.add_argument("--backbone-epochs", type=int, default=200)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)

    # soft prototype
    parser.add_argument("--soft-epochs", type=int, default=100)
    parser.add_argument("--soft-lr", type=float, default=1e-2)
    parser.add_argument("--soft-lambda-balance", type=float, default=1.0)
    parser.add_argument("--soft-mode", type=str, default="top1", choices=["top1", "avg", "sharpened_avg"])
    parser.add_argument("--soft-gamma", type=float, default=1.0)

    # sgcp
    parser.add_argument("--sgcp-epochs", type=int, default=200)
    parser.add_argument("--sgcp-lr", type=float, default=1e-3)
    parser.add_argument("--num-score-bins", type=int, default=20)
    parser.add_argument("--score-bin-edges", type=str, default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--hist-smoothing", type=float, default=1e-3)

    # sweep
    parser.add_argument("--grid-json", type=str, default="grid_sgcp.json")
    parser.add_argument(
        "--sweep-name",
        type=str,
        default="default_sweep",
        help="Used in aggregated summary filenames.",
    )
    parser.set_defaults(skip_existing=True)

    return parser

def config_from_args(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    cfg = default_cfg()

    cfg["experiment"].update(
        {
            "alpha": args.alpha,
            "hard_cluster_seed": getattr(args, "hard_cluster_seed", args.seed),
            "outdir": args.outdir,
            "methods": args.methods,
            "seed": args.seed,
            "cuda": args.cuda,
        }
    )
    cfg["dataset"].update(
        {
            "n_tra_cal": args.n_tra_cal,
            "test_samples": args.test_samples,
            "color_blue_prob": args.color_blue_prob,
            "group1_prob_1": args.group1_prob_1,
            "group2_prob_1": args.group2_prob_1,
            "delta1": args.delta1,
            "delta0": args.delta0,
            "n_nonsensitive": args.n_nonsensitive,
            "batch_size": args.batch_size,
            "dataset_mode": args.dataset_mode,
        }
    )

    cfg["model"].update(
        {
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "num_prototypes": args.num_prototypes,
            "num_classes": args.num_classes,
            "temperature": args.temperature,
            "stochastic_hidden_dim": args.stochastic_hidden_dim,
            "stochastic_num_hidden": args.stochastic_num_hidden,
            "min_sig": args.min_sig,
        }
    )

    cfg["backbone_train"].update(
        {
            "epochs": args.backbone_epochs,
            "lr": args.backbone_lr,
        }
    )

    cfg["sgcp_train"].update(
        {
            "epochs": args.sgcp_epochs,
            "lr": args.sgcp_lr,
            "num_score_bins": args.num_score_bins,
            "score_bin_edges": args.score_bin_edges,
            "hist_smoothing": args.hist_smoothing,
        }
    )

    return cfg


def save_run_results(
    results,
    cfg: Dict[str, Dict[str, Any]],
    dataset_name,
    outdir="results",
):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    decimals = get_result_decimals()

    def _round_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
        metric_cols = [c for c in df.columns if c.startswith("metric.")]
        for c in metric_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].round(decimals)
        return df

    def _reorder_covgap_columns(df: pd.DataFrame) -> pd.DataFrame:
        anchor = "metric.age_set_sizes"
        preferred = [
            "metric.covgap",
            "metric.covgap_color",
            "metric.covgap_age",
            "metric.covgap_region",
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
    base_info.update(flatten_config_dict("experiment", cfg["experiment"]))
    base_info.update(flatten_config_dict("dataset", cfg["dataset"]))
    base_info.update(flatten_config_dict("model", cfg["model"]))
    base_info.update(flatten_config_dict("backbone_train", cfg["backbone_train"]))
    base_info.update(flatten_config_dict("sgcp_train", cfg["sgcp_train"]))
    base_info["dataset_name"] = dataset_name

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

    # run_csv_path = os.path.join(outdir, f"{run_name}.csv")
    # run_df.to_csv(run_csv_path, index=False)

    master_csv_path = os.path.join(outdir, "all_runs_summary.csv")
    if os.path.exists(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
        merged = pd.concat([master_df, run_df], ignore_index=True)
        merged = _reorder_covgap_columns(merged)
        merged = _round_metric_columns(merged)
        merged.to_csv(master_csv_path, index=False)
    else:
        run_df.to_csv(master_csv_path, index=False)

    # print(f"\nSaved per-run JSON: {json_path}")
    # print(f"Saved per-run CSV : {run_csv_path}")
    print(f"Updated summary   : {master_csv_path}\n")

    return run_df, master_csv_path


def run_name_for_cfg(cfg: Dict[str, Dict[str, Any]], dataset_name: str) -> str:
    return (
        f"{dataset_name}"
        f"_seed{cfg['experiment']['seed']}"
        f"_trca{cfg['dataset']['n_tra_cal']}"
        f"_te{cfg['dataset']['test_samples']}"
        f"_bblr{cfg['backbone_train']['lr']}"
        f"_bbep{cfg['backbone_train']['epochs']}"
        f"_sgcplr{cfg['sgcp_train']['lr']}"
        f"_sgcpep{cfg['sgcp_train']['epochs']}"
        f"_kl{cfg['sgcp_train']['beta_kl']}"
        f"_bal{cfg['sgcp_train']['lambda_balance']}"
        f"_nll{cfg['sgcp_train']['lambda_score']}"
        f"_lats{cfg['sgcp_train']['train_latent_samples']}"
        f"_evalL{cfg['sgcp_train']['eval_latent_samples']}"
        f"_bins{cfg['sgcp_train']['num_score_bins']}"
        f"_edges{cfg['sgcp_train']['score_bin_edges']}"
        f"_smooth{cfg['sgcp_train']['hist_smoothing']}"
        f"_proto{cfg['model']['num_prototypes']}"
        f"_latent{cfg['model']['latent_dim']}"
    )


def dataset_name_for_cfg(cfg: Dict[str, Dict[str, Any]]) -> str:
    return "two_sensitive" if cfg["dataset"]["dataset_mode"] == "two_sensitive" else "single_sensitive"


def run_csv_path_for_cfg(cfg: Dict[str, Dict[str, Any]], dataset_name: str) -> str:
    return os.path.join(cfg["experiment"]["outdir"], f"{run_name_for_cfg(cfg, dataset_name)}.csv")


def build_dataset_and_loaders(data_cfg: Dict[str, Any], model_cfg: Dict[str, Any], seed: int):
    
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
        # input_dim = 4 + data_cfg["n_nonsensitive"]
        dataset_name = "two_sensitive"
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
        # input_dim = 3 + data_cfg["n_nonsensitive"]
        dataset_name = "single_sensitive"

    return train_loader, cal_loader, test_loader, dataset_name, syn_cfg


# =========================
# Core run
# =========================

def run_one_experiment(
    cfg: Dict[str, Dict[str, Any]],
):
    exp_cfg = SimpleNamespace(**cfg["experiment"])
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


    train_loader, cal_loader, test_loader, dataset_name, syn_cfg = build_dataset_and_loaders(
        data_cfg=cfg["dataset"],
        model_cfg=cfg["model"],
        seed=exp_cfg.seed,
    )

    print(f"[runtime] dataset_name: {dataset_name}")
    print(f"[runtime] synthetic_cfg: {vars(syn_cfg)}\n")

    backbone = Backbone(
        input_dim=next(iter(train_loader))[0].shape[1],
        hidden_dim=model_cfg.hidden_dim,
        num_classes=model_cfg.num_classes,
    )

    backbone = train_backbone(
        backbone,
        train_loader,
        epochs=cfg["backbone_train"]["epochs"],
        lr=cfg["backbone_train"]["lr"],
        device=device,
    )


    results = evaluate_sgcp(
        backbone=backbone,
        train_loader=train_loader,
        cal_loader=cal_loader,
        test_loader=test_loader,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        sgcp_cfg=sgcp_cfg,
        device=device,
        )

    for title, metrics in results.items():
        print(f"\n=== {title} ===")
        print(metrics)

    run_df, master_csv_path = save_run_results(
        results=results,
        cfg=cfg,
        dataset_name=dataset_name,
        outdir=exp_cfg.outdir,
    )

    return {
        "run_df": run_df,
        # "json_path": json_path,
        # "run_csv_path": run_csv_path,
        "master_csv_path": master_csv_path,
        "dataset_name": dataset_name,
    }


# =========================
# Main
# =========================

def main(args=None):
    parser = build_parser()
    parsed_args = parser.parse_args(args=args)

    base_cfg = config_from_args(parsed_args)
    base_exp_cfg = base_cfg["experiment"]

    # grid
    grid = load_grid_json(parsed_args.grid_json)
    grid_seed_list, grid_seed_key = pop_seed_sweep_from_grid(grid)
    grid_points = expand_grid(grid)

    # seed list
    seed_list = resolve_seed_list(
        cli_seeds=parsed_args.seeds,
        default_seed=base_exp_cfg["seed"],
        grid_seed_list=grid_seed_list,
        grid_seed_key=grid_seed_key,
    )

    Path(base_exp_cfg["outdir"]).mkdir(parents=True, exist_ok=True)

    print(f"[sweep] seeds      : {seed_list}")
    print(f"[sweep] grid size  : {len(grid_points)}")
    print(f"[sweep] total runs : {len(seed_list) * len(grid_points)}")
    if grid:
        print(f"[sweep] grid keys  : {list(grid.keys())}")
    print()

    all_run_dfs = []
    run_counter = 0
    total_runs = len(seed_list) * len(grid_points)

    run_iter = [
        (grid_idx, overrides, seed)
        for grid_idx, overrides in enumerate(grid_points)
        for seed in seed_list
    ]

    pbar = tqdm(run_iter, total=total_runs, desc="Hyperparameter Sweep", dynamic_ncols=True)

    for run_idx, (grid_idx, overrides, seed) in enumerate(pbar, start=1):
        cfg = clone_cfg(base_cfg)
        cfg["experiment"]["seed"] = seed
        apply_overrides(cfg=cfg, override_dict=overrides)

        # dataset_name = dataset_name_for_cfg(cfg)

        pbar.set_postfix({
            "seed": seed,
            "proto": cfg["model"]["num_prototypes"],
            "latent": cfg["model"]["latent_dim"],
        })

        pbar.write(f"[run {run_idx}/{total_runs}] seed={seed}, overrides={overrides}")
        out = run_one_experiment(cfg=cfg)
        all_run_dfs.append(out["run_df"])



    combined_df = pd.concat(all_run_dfs, ignore_index=True)

    sweep_long_path = os.path.join(base_exp_cfg["outdir"], f"{parsed_args.sweep_name}_combined_long.csv")
    combined_df = combined_df.copy()
    combined_df = _round_metric_columns(combined_df)
    combined_df.to_csv(sweep_long_path, index=False)

    agg_df = aggregate_summary_from_df(combined_df)
    sweep_agg_path = os.path.join(base_exp_cfg["outdir"], f"{parsed_args.sweep_name}_grid_summary_mean_std.csv")
    agg_df = agg_df.copy()
    agg_df = _round_metric_columns(agg_df)
    agg_df.to_csv(sweep_agg_path, index=False)

    print("\n" + "#" * 100)
    print("Sweep finished.")
    print(f"Combined long results : {sweep_long_path}")
    print(f"Aggregated mean/std   : {sweep_agg_path}")
    print("#" * 100 + "\n")


if __name__ == "__main__":
    main()
