import torch

from CP_fairness.sgcp_flow import StochasticAssignment
from util.eval_tool import calibrate_sgcp_quantile, evaluate_sg_cp, fit_sgcp
from util.train_tool import train_stochastic_assignment


def _sync_if_cuda(device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_sgcp(backbone, train_loader, train_cal_loader, cal_loader, test_loader, exp_cfg, model_cfg, sgcp_cfg, device):
    alpha = exp_cfg.alpha
    sg_assign = StochasticAssignment(
        backbone=backbone,
        latent_dim=model_cfg.latent_dim,
        num_prototypes=model_cfg.num_prototypes,
        temperature=model_cfg.temperature,
        stochastic_hidden_dim=model_cfg.stochastic_hidden_dim,
        stochastic_num_hidden=model_cfg.stochastic_num_hidden,
        prior_in_dim=model_cfg.num_classes,
        min_sig=model_cfg.min_sig,
    )
    sg_assign = train_stochastic_assignment(
        sg_assign,
        train_loader,
        epochs=sgcp_cfg.epochs,
        lr=sgcp_cfg.lr,
        lambda_nll=sgcp_cfg.lambda_nll,
        lambda_balance=sgcp_cfg.lambda_balance,
        lambda_rank=sgcp_cfg.lambda_rank,
        n_latent_samples=sgcp_cfg.train_latent_samples,
        device=device,
    )

    sg_cp = fit_sgcp(
        backbone=backbone,
        assign_model=sg_assign,
        transform_loader=train_cal_loader,
        device=device,
        n_latent_samples=sgcp_cfg.eval_latent_samples,
    )

    sg_cp = calibrate_sgcp_quantile(
        backbone=backbone,
        assign_model=sg_assign,
        sgcp_obj=sg_cp,
        cal_loader=cal_loader,
        alpha=alpha,
        device=device,
        n_latent_samples=sgcp_cfg.eval_latent_samples,
    )

    results = {}
    results["SGCP"] = evaluate_sg_cp(
        backbone,
        sg_assign,
        sg_cp,
        test_loader,
        dataset_name=exp_cfg.dataset_mode,
        device=device,
        n_latent_samples=sgcp_cfg.eval_latent_samples,
        alpha=alpha,
    )
    return results, sg_assign, sg_cp
