import argparse
from dataclasses import asdict, dataclass

import torch
from data.synthetic import build_dataloaders

from eval import evaluate_all_methods
from util.utils import set_seed
from util.train_tool import *

from SelectiveCI_fairness.cp_obj import *
from SelectiveCI_fairness.sls_flow import Backbone


# =========================
# config 
# =========================

@dataclass
class DatasetConfig:
    train_samples: int = 4000
    cal_samples: int = 1000
    test_samples: int = 1000
    color_blue_prob: float = 0.10
    n_nonsensitive: int = 6
    train_seed: int = 1
    cal_seed: int = 2
    test_seed: int = 3
    batch_size: int = 128

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    feature_dim: int = 32
    latent_dim: int = 8
    num_prototypes: int = 8
    num_classes: int = 6
    temperature: float = 1.0
    stochastic_hidden_dim: int = 64
    stochastic_num_hidden: int = 2
    min_sig: float = 1e-3


@dataclass
class BackboneTrainConfig:
    epochs: int = 200
    lr: float = 1e-3


@dataclass
class SoftProtoTrainConfig:
    epochs: int = 100
    lr: float = 1e-2
    lambda_balance: float = 1.0
    mode: str = "top1"
    gamma: float = 1.0


@dataclass
class SLSTrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    beta_kl: float = 1e-6
    lambda_balance: float = 1e-2
    lambda_score: float = 0.5
    lambda_tail: float = 1.0
    lambda_miss: float = 1.0
    lambda_difficulty: float = 0.5
    lambda_proto_risk: float = 0.05
    tail_quantile: float = 0.9
    train_latent_samples: int = 8
    eval_latent_samples: int = 10
    num_bins: int = 5
    min_bin_n: int = 30


@dataclass
class ExperimentConfig:
    seed: int = 42
    alpha: float = 0.1
    hard_cluster_seed: int = 42
    cuda: str = "0"
    print_config: bool = True
    run_afcp_adaptive: bool = False
    afcp_ttest_delta: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--print-config", action="store_true", default=True)
    parser.add_argument("--no-print-config", dest="print_config", action="store_false")
    parser.add_argument("--run-afcp-adaptive", action="store_true", help="Run the very slow AFCP Adaptive Selection baseline")
    parser.add_argument("--afcp-ttest-delta", type=float, default=None)

    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--cal-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--color-blue-prob", type=float, default=0.10)
    parser.add_argument("--n-nonsensitive", type=int, default=6)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--cal-seed", type=int, default=2)
    parser.add_argument("--test-seed", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stochastic-hidden-dim", type=int, default=64)
    parser.add_argument("--stochastic-num-hidden", type=int, default=2)
    parser.add_argument("--min-sig", type=float, default=1e-3)

    parser.add_argument("--backbone-epochs", type=int, default=200)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)

    parser.add_argument("--soft-epochs", type=int, default=100)
    parser.add_argument("--soft-lr", type=float, default=1e-2)
    parser.add_argument("--soft-lambda-balance", type=float, default=1.0)
    parser.add_argument("--soft-mode", type=str, default="top1", choices=["top1", "avg", "sharpened_avg"])
    parser.add_argument("--soft-gamma", type=float, default=1.0)

    parser.add_argument("--sls-epochs", type=int, default=200)
    parser.add_argument("--sls-lr", type=float, default=1e-3)
    parser.add_argument("--beta-kl", type=float, default=1e-6)
    parser.add_argument("--lambda-balance", type=float, default=1e-2)
    parser.add_argument("--lambda-score", type=float, default=0.5)
    parser.add_argument("--lambda-tail", type=float, default=1.0)
    parser.add_argument("--lambda-miss", type=float, default=1.0)
    parser.add_argument("--lambda-difficulty", type=float, default=0.5)
    parser.add_argument("--lambda-proto-risk", type=float, default=0.05)
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--train-latent-samples", type=int, default=8)
    parser.add_argument("--eval-latent-samples", type=int, default=10)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--min-bin-n", type=int, default=30)
    parser.add_argument("--hard-cluster-seed", type=int, default=42)

    return parser


def config_from_args(args: argparse.Namespace):
    exp_cfg = ExperimentConfig(
        seed=args.seed,
        alpha=args.alpha,
        hard_cluster_seed=args.hard_cluster_seed,
        cuda=args.cuda,
        print_config=args.print_config,
        run_afcp_adaptive=args.run_afcp_adaptive,
        afcp_ttest_delta=args.afcp_ttest_delta,
    )
    data_cfg = DatasetConfig(
        train_samples=args.train_samples,
        cal_samples=args.cal_samples,
        test_samples=args.test_samples,
        color_blue_prob=args.color_blue_prob,
        n_nonsensitive=args.n_nonsensitive,
        train_seed=args.train_seed,
        cal_seed=args.cal_seed,
        test_seed=args.test_seed,
        batch_size=args.batch_size,
    )
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        num_prototypes=args.num_prototypes,
        num_classes=args.num_classes,
        temperature=args.temperature,
        stochastic_hidden_dim=args.stochastic_hidden_dim,
        stochastic_num_hidden=args.stochastic_num_hidden,
        min_sig=args.min_sig,
    )
    backbone_cfg = BackboneTrainConfig(epochs=args.backbone_epochs, lr=args.backbone_lr)
    soft_cfg = SoftProtoTrainConfig(
        epochs=args.soft_epochs,
        lr=args.soft_lr,
        lambda_balance=args.soft_lambda_balance,
        mode=args.soft_mode,
        gamma=args.soft_gamma,
    )
    sls_cfg = SLSTrainConfig(
        epochs=args.sls_epochs,
        lr=args.sls_lr,
        beta_kl=args.beta_kl,
        lambda_balance=args.lambda_balance,
        lambda_score=args.lambda_score,
        lambda_tail=args.lambda_tail,
        lambda_miss=args.lambda_miss,
        lambda_difficulty=args.lambda_difficulty,
        lambda_proto_risk=args.lambda_proto_risk,
        tail_quantile=args.tail_quantile,
        train_latent_samples=args.train_latent_samples,
        eval_latent_samples=args.eval_latent_samples,
        num_bins=args.num_bins,
        min_bin_n=args.min_bin_n,
    )
    return exp_cfg, data_cfg, model_cfg, backbone_cfg, soft_cfg, sls_cfg


def print_configs(**configs):
    print("\n=== Config ===")
    for name, cfg in configs.items():
        print(f"[{name}]")
        for k, v in asdict(cfg).items():
            print(f"  {k}: {v}")


def main(args=None):
    parser = build_parser()
    parsed_args = parser.parse_args(args=args)
    exp_cfg, data_cfg, model_cfg, backbone_cfg, soft_cfg, sls_cfg = config_from_args(parsed_args)

    set_seed(exp_cfg.seed)
    device = torch.device("cuda:" + exp_cfg.cuda if torch.cuda.is_available() else "cpu")

    if exp_cfg.print_config:
        print_configs(
            experiment=exp_cfg,
            dataset=data_cfg,
            model=model_cfg,
            backbone_train=backbone_cfg,
            softproto_train=soft_cfg,
            sls_train=sls_cfg,
        )
        print(f"[runtime] device: {device}\n")

    train_loader, cal_loader, test_loader = build_dataloaders(data_cfg)

    input_dim = 3 + data_cfg.n_nonsensitive
    backbone = Backbone(
        input_dim=input_dim,
        hidden_dim=model_cfg.hidden_dim,
        feature_dim=model_cfg.feature_dim,
        num_classes=model_cfg.num_classes,
    )
    backbone = train_backbone(
        backbone,
        train_loader,
        epochs=backbone_cfg.epochs,
        lr=backbone_cfg.lr,
        device=device,
    )

    results = evaluate_all_methods(
        backbone=backbone,
        train_loader=train_loader,
        cal_loader=cal_loader,
        test_loader=test_loader,
        exp_cfg=exp_cfg,
        model_cfg=model_cfg,
        soft_cfg=soft_cfg,
        sls_cfg=sls_cfg,
        device=device,
    )

    for title, metrics in results.items():
        print(f"\n=== {title} ===")
        print(metrics)


if __name__ == "__main__":
    main()
