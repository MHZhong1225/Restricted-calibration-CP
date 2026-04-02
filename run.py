import argparse
import os
from types import SimpleNamespace
from tqdm import tqdm

from stoCP import build_parser, config_from_args, run_experiment
from util.helper import (
    load_grid_json, expand_grid, clone_cfg, 
    pop_seed_sweep_from_grid, apply_overrides, resolve_seed_list
)

def main():
    parser = build_parser()
    args = parser.parse_args()
    base_cfg = config_from_args(args)

    # (Grid Sweep
    if args.grid_json and os.path.exists(args.grid_json):
        print(f"\n[Mode] Sweep Mode: Loading config from {args.grid_json}")
        grid = load_grid_json(args.grid_json)
        expanded = expand_grid(grid)

        for run_idx, override_dict in enumerate(tqdm(expanded, desc="Hyperparameter Sweep")):
            run_cfg = clone_cfg(base_cfg)
            grid_seeds, grid_seed_key = pop_seed_sweep_from_grid(override_dict)
            apply_overrides(run_cfg, override_dict)
            
            cli_seeds = getattr(args, "seeds", None)
            seeds_to_run = resolve_seed_list(cli_seeds, base_cfg["experiment"]["seed"], grid_seeds, grid_seed_key)

            for s in seeds_to_run:
                run_cfg["experiment"]["seed"] = s
                exp_cfg = SimpleNamespace(**run_cfg["experiment"])
                exp_cfg.dataset_mode = run_cfg.get("dataset", {}).get("dataset_mode")
                run_experiment(cfg=run_cfg, exp_cfg=exp_cfg)

    # Single Run
    else:
        print("\n[Mode] Single Experiment Mode: Using command line args only.")
        cli_seeds = getattr(args, "seeds", None)
        seeds_to_run = cli_seeds if cli_seeds else [base_cfg["experiment"]["seed"]]

        for s in seeds_to_run:
            run_cfg = clone_cfg(base_cfg)
            run_cfg["experiment"]["seed"] = s
            exp_cfg = SimpleNamespace(**run_cfg["experiment"])
            exp_cfg.dataset_mode = run_cfg.get("dataset", {}).get("dataset_mode")
            run_experiment(cfg=run_cfg, exp_cfg=exp_cfg)

if __name__ == "__main__":
    main()
