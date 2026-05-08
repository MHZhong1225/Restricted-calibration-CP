from sgcp import (
    apply_overrides,
    build_parser,
    config_from_args,
    expand_grid,
    load_grid_json,
    run_experiment,
)


def main() -> None:
    parser = build_parser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of random seeds to run. If not provided, use --seed.",
    )
    args = parser.parse_args()

    grid = load_grid_json(args.grid_json)
    grid_seeds = grid.get("seeds", grid.get("seed"))
    base_cfg = config_from_args(args)
    seeds_to_run = args.seeds if args.seeds else (grid_seeds if grid_seeds else [base_cfg["experiment"]["seed"]])

    for override in expand_grid(grid):
        for seed in seeds_to_run:
            run_cfg = config_from_args(args)
            apply_overrides(run_cfg, override)
            run_cfg["experiment"]["seed"] = int(seed)
            dataset = run_cfg["dataset"]["dataset_mode"]
            print(f"\n[Run] dataset={dataset} seed={seed}")
            run_experiment(run_cfg)


if __name__ == "__main__":
    main()
