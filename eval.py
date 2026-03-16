

import numpy as np
import torch

from SelectiveCI_fairness.methods import ExhaustiveSelection, PartialSelection, MarginalSelection, AFCPAdaptiveSelection, FaReG
from SelectiveCI_fairness.sls_flow import StochasticAssignment, SoftPrototypeAssignment

from util.eval_tool import (
    evaluate_global_cp,
    evaluate_fixed_group_cp,
    calibrate_global_cp,
    calibrate_fixed_group_cp,
    calibrate_hard_cluster_cp,
    evaluate_hard_cluster_cp,
    calibrate_soft_prototype_cp,
    calibrate_sls_cp,
    evaluate_soft_prototype_cp,
    evaluate_sls_cp,
)
from util.utils import loader_to_numpy, extract_all, conformal_quantile, prediction_set_from_probs_and_thresholds
from util.train_tool import *

def _ensure_nonempty(pred_sets, probs):
    out = []
    for i, s in enumerate(pred_sets):
        if len(s) == 0:
            out.append([int(np.argmax(probs[i]))])
        else:
            out.append(s)
    return out


def prediction_sets_to_metrics(test_np, C_sets):
    y = test_np['y']
    color = test_np['color']
    age = test_np['age']
    region = test_np['region']
    cover = np.array([int(int(y[i]) in set(C_sets[i])) for i in range(len(y))], dtype=float)
    size = np.array([len(C_sets[i]) for i in range(len(y))], dtype=float)
    def grp(vals, key):
        out = {}
        for k in sorted(np.unique(key)):
            m = key == k
            out[int(k)] = float(vals[m].mean()) if m.any() else float('nan')
        return out
    return {
        'overall_coverage': float(cover.mean()),
        'avg_set_size': float(size.mean()),
        'color_coverages': grp(cover, color),
        'color_set_sizes': grp(size, color),
        'age_coverages': grp(cover, age),
        'age_set_sizes': grp(size, age),
        'region_coverages': grp(cover, region),
        'region_set_sizes': grp(size, region),
        'blue_coverage': float(cover[color == 1].mean()),
        'blue_avg_set_size': float(size[color == 1].mean()),
    }


@torch.no_grad()
def evaluate_fareg_cp(backbone, fareg, cal_loader, test_loader, alpha, device="cpu"):
    cal_data = extract_all(backbone, cal_loader, device=device)
    test_data = extract_all(backbone, test_loader, device=device)

    cal_feats = torch.tensor(cal_data["feats"], device=device, dtype=torch.float32)
    test_feats = torch.tensor(test_data["feats"], device=device, dtype=torch.float32)
    cal_prob, _, _ = fareg(cal_feats)
    test_prob, _, _ = fareg(test_feats)
    cal_prob = cal_prob.squeeze(-1)
    test_prob = test_prob.squeeze(-1)

    global_q = conformal_quantile(cal_data["scores"], alpha)

    best_gamma = 0.0
    best_size = float("inf")
    target_cov = 1.0 - alpha
    cal_prob_np = cal_prob.detach().cpu().numpy()
    for gamma in np.linspace(0.0, 1.0, 11):
        thr = global_q + gamma * (0.5 - cal_prob_np)
        pred_sets = prediction_set_from_probs_and_thresholds(cal_data["probs"], thr)
        pred_sets = _ensure_nonempty(pred_sets, cal_data["probs"])
        cover = np.array([int(cal_data["y"][i] in pred_sets[i]) for i in range(len(pred_sets))], dtype=float).mean()
        sizes = np.array([len(s) for s in pred_sets], dtype=float).mean()
        if cover >= target_cov and sizes < best_size:
            best_size = sizes
            best_gamma = float(gamma)

    test_prob_np = test_prob.detach().cpu().numpy()
    thr_test = global_q + best_gamma * (0.5 - test_prob_np)
    pred_sets_test = prediction_set_from_probs_and_thresholds(test_data["probs"], thr_test)
    pred_sets_test = _ensure_nonempty(pred_sets_test, test_data["probs"])
    return prediction_sets_to_metrics(loader_to_numpy(test_loader), pred_sets_test)


def evaluate_all_methods(backbone, train_loader, cal_loader, test_loader, exp_cfg, model_cfg, soft_cfg, sls_cfg, device):
    alpha = exp_cfg.alpha

    marginal_cp = calibrate_global_cp(backbone, cal_loader, alpha=alpha, device=device)
    partial_color_cp = calibrate_fixed_group_cp(
        backbone,
        cal_loader,
        key_fn=lambda d: list(d["color"]),
        alpha=alpha,
        device=device,
    )
    exhaustive_cp = calibrate_fixed_group_cp(
        backbone,
        cal_loader,
        key_fn=lambda d: list(zip(d["color"], d["age"], d["region"])),
        alpha=alpha,
        device=device,
    )

    results = {
        "Marginal (paper baseline)": evaluate_global_cp(backbone, marginal_cp, test_loader, device=device),
        "Partial(Color) (paper baseline)": evaluate_fixed_group_cp(
            backbone,
            partial_color_cp,
            test_loader,
            key_fn=lambda d: list(d["color"]),
            device=device,
        ),
        "Exhaustive(Color,Age,Region) (paper baseline)": evaluate_fixed_group_cp(
            backbone,
            exhaustive_cp,
            test_loader,
            key_fn=lambda d: list(zip(d["color"], d["age"], d["region"])),
            device=device,
        ),
    }

    cal_np = loader_to_numpy(cal_loader)
    test_np = loader_to_numpy(test_loader)

    marg = MarginalSelection(alpha=alpha)
    partial = PartialSelection(alpha=alpha)
    exhaustive = ExhaustiveSelection(alpha=alpha)

    results["Marginal Selection"] = prediction_sets_to_metrics(
        test_np,
        marg.multiclass_classification(
            test_np['x'], cal_np['x'], cal_np['y'], backbone, conditional=True, device=device
        ),
    )
    results["Partial Selection (Color)"] = prediction_sets_to_metrics(
        test_np,
        partial.multiclass_classification(
            test_np['x'], cal_np['x'], cal_np['y'], backbone,
            sensitive_atts_idx=[0], conditional=True, device=device
        ),
    )
    results["Exhaustive Selection (Color,Age,Region)"] = prediction_sets_to_metrics(
        test_np,
        exhaustive.multiclass_classification(
            test_np['x'], cal_np['x'], cal_np['y'], backbone,
            sensitive_atts_idx=[0,1,2], conditional=True, device=device
        ),
    )
    if exp_cfg.run_afcp_adaptive:
        afcp_adaptive = AFCPAdaptiveSelection(alpha=alpha, ttest_delta=exp_cfg.afcp_ttest_delta)
        adaptive_sets, adaptive_k = afcp_adaptive.multiclass_classification(
            cal_np['x'], cal_np['y'], test_np['x'], backbone, att_idx=[0, 1, 2], conditional=False, device=device
        )
        results["AFCP Adaptive Selection"] = prediction_sets_to_metrics(test_np, adaptive_sets)
        selected_count = sum(1 for ks in adaptive_k if len(ks) > 0)
        results["AFCP Adaptive Selection"]["selected_attribute_rate"] = float(selected_count / max(len(adaptive_k), 1))

    hard_cp = calibrate_hard_cluster_cp(
        backbone,
        cal_loader,
        alpha=alpha,
        K=model_cfg.num_prototypes,
        device=device,
        seed=exp_cfg.hard_cluster_seed,
    )
    results["Hard Cluster CP"] = evaluate_hard_cluster_cp(backbone, hard_cp, test_loader, device=device)

    soft_assign = SoftPrototypeAssignment(
        backbone=backbone,
        feature_dim=model_cfg.feature_dim,
        num_prototypes=model_cfg.num_prototypes,
        temperature=model_cfg.temperature,
    )
    soft_assign = train_soft_prototype_assignment(
        soft_assign,
        train_loader,
        epochs=soft_cfg.epochs,
        lr=soft_cfg.lr,
        lambda_balance=soft_cfg.lambda_balance,
        device=device,
    )
    soft_cp = calibrate_soft_prototype_cp(
        backbone,
        soft_assign,
        cal_loader,
        alpha=alpha,
        device=device,
        mode=soft_cfg.mode,
        gamma=soft_cfg.gamma,
    )
    results["Soft Prototype CP"] = evaluate_soft_prototype_cp(
        backbone,
        soft_assign,
        soft_cp,
        test_loader,
        device=device,
    )

    sls_assign = StochasticAssignment(
        backbone=backbone,
        feature_dim=model_cfg.feature_dim,
        latent_dim=model_cfg.latent_dim,
        num_prototypes=model_cfg.num_prototypes,
        temperature=model_cfg.temperature,
        stochastic_hidden_dim=model_cfg.stochastic_hidden_dim,
        stochastic_num_hidden=model_cfg.stochastic_num_hidden,
        prior_in_dim=model_cfg.num_classes,
        min_sig=model_cfg.min_sig,
    )
    sls_assign = train_stochastic_assignment(
        sls_assign,
        train_loader,
        backbone=backbone,
        alpha=alpha,
        epochs=sls_cfg.epochs,
        lr=sls_cfg.lr,
        beta_kl=sls_cfg.beta_kl,
        lambda_balance=sls_cfg.lambda_balance,
        lambda_score=sls_cfg.lambda_score,
        lambda_tail=sls_cfg.lambda_tail,
        lambda_miss=sls_cfg.lambda_miss,
        lambda_difficulty=sls_cfg.lambda_difficulty,
        lambda_proto_risk=sls_cfg.lambda_proto_risk,
        tail_quantile=sls_cfg.tail_quantile,
        n_latent_samples=sls_cfg.train_latent_samples,
        device=device,
    )
    sls_cp = calibrate_sls_cp(
        backbone,
        sls_assign,
        cal_loader,
        alpha=alpha,
        device=device,
        n_latent_samples=sls_cfg.eval_latent_samples,
        num_bins=sls_cfg.num_bins,
        min_bin_n=sls_cfg.min_bin_n,
    )
    fareg = FaReG(
        num_features=model_cfg.feature_dim,
        delta=alpha,
        device=device,
        hidden_dim=64,
        latent_dim=32,
    ).to(device)


    results["FaReG (difficulty-conditioned CP)"] = evaluate_fareg_cp(
        backbone=backbone,
        fareg=fareg,
        cal_loader=cal_loader,
        test_loader=test_loader,
        alpha=alpha,
        device=device,
    )

    results["SLS CP"] = evaluate_sls_cp(
        backbone,
        sls_assign,
        sls_cp,
        test_loader,
        device=device,
        n_latent_samples=sls_cfg.eval_latent_samples,
    )


    return results
