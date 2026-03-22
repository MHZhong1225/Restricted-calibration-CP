import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from SelectiveCI_fairness.methods import ExhaustiveSelection, PartialSelection, MarginalSelection, AFCPAdaptiveSelection
from SelectiveCI_fairness.sgcp_flow import StochasticAssignment, SoftPrototypeAssignment

from FaReG.Group_fairness.cp import Marginal_Fairness, FaReG_Fairness
from FaReG.Group_fairness.networks import FaReG as FaReGNet


from util.eval_tool import (
    evaluate_global_cp,
    evaluate_fixed_group_cp,
    calibrate_global_cp,
    calibrate_fixed_group_cp,
    calibrate_hard_cluster_cp,
    evaluate_hard_cluster_cp,
    calibrate_soft_prototype_cp,
    calibrate_sgcp,
    evaluate_soft_prototype_cp,
    evaluate_sg_cp,
)
from util.utils import loader_to_numpy
from util.train_tool import *

def prediction_sets_to_metrics(test_np, C_sets, alpha):
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
    def cov_gap_from_coverages(coverages):
        vals = [v for v in coverages.values() if not np.isnan(v)]
        if len(vals) == 0:
            return float("nan")
        target = 1.0 - float(alpha)
        return float(np.mean(np.abs(np.asarray(vals, dtype=float) - target)))
    def cov_gap_from_keys(keys):
        uniq = np.unique(keys)
        if len(uniq) == 0:
            return float("nan")
        target = 1.0 - float(alpha)
        gaps = []
        for k in uniq:
            m = keys == k
            if not m.any():
                continue
            gaps.append(abs(float(cover[m].mean()) - target))
        if len(gaps) == 0:
            return float("nan")
        return float(np.mean(np.asarray(gaps, dtype=float)))
    color_cov = grp(cover, color)
    age_cov = grp(cover, age)
    region_cov = grp(cover, region)
    joint_keys = np.asarray([f"{int(c)}|{int(a)}|{int(r)}" for c, a, r in zip(color, age, region)], dtype=object)
    covgap_color = cov_gap_from_coverages(color_cov)
    covgap_age = cov_gap_from_coverages(age_cov)
    covgap_region = cov_gap_from_coverages(region_cov)
    covgap = cov_gap_from_keys(joint_keys)
    return {
        'overall_coverage': float(cover.mean()),
        'avg_set_size': float(size.mean()),
        'color_coverages': color_cov,
        'color_set_sizes': grp(size, color),
        'age_coverages': age_cov,
        'age_set_sizes': grp(size, age),
        'region_coverages': region_cov,
        'region_set_sizes': grp(size, region),
        'blue_coverage': float(cover[color == 1].mean()),
        'blue_avg_set_size': float(size[color == 1].mean()),
        'covgap': covgap,
        'covgap_color': covgap_color,
        'covgap_age': covgap_age,
        'covgap_region': covgap_region,
    }


def evaluate_all_methods(backbone, train_loader, cal_loader, test_loader, exp_cfg, model_cfg, soft_cfg, device):
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
        "Marginal Selection": evaluate_global_cp(backbone, marginal_cp, test_loader, device=device, alpha=alpha),
        "Partial Selection (Color)": evaluate_fixed_group_cp(
            backbone,
            partial_color_cp,
            test_loader,
            key_fn=lambda d: list(d["color"]),
            device=device,
            alpha=alpha,
        ),
        "Exhaustive Selection (Color,Age,Region)": evaluate_fixed_group_cp(
            backbone,
            exhaustive_cp,
            test_loader,
            key_fn=lambda d: list(zip(d["color"], d["age"], d["region"])),
            device=device,
            alpha=alpha,
        ),
    }

    if getattr(exp_cfg, "run_afcp_adaptive", False):
        cal_np = loader_to_numpy(cal_loader)
        test_np = loader_to_numpy(test_loader)
        X_calib = cal_np["x"].astype(np.float32)
        Y_calib = cal_np["y"].astype(int)
        X_test = test_np["x"].astype(np.float32)

        batch0 = next(iter(cal_loader))
        d = int(X_calib.shape[1])
        n_check = min(256, len(X_calib))

        group_X_calib = None
        group_X_test = None

        if len(batch0) == 5:
            att_idx_emb = [d - 1, d - 2, d - 3]
            embedded_ok = False
            if n_check > 0:
                embedded_ok = float(np.mean(X_calib[:n_check, att_idx_emb[0]].astype(int) == cal_np["color"][:n_check].astype(int))) >= 0.95
            if embedded_ok:
                att_idx = att_idx_emb
            else:
                group_X_calib = np.column_stack([cal_np["color"], cal_np["age"], cal_np["region"]]).astype(int)
                group_X_test = np.column_stack([test_np["color"], test_np["age"], test_np["region"]]).astype(int)
                att_idx = [0, 1, 2]
        elif len(batch0) == 6:
            att_idx = [d - 1, d - 2, d - 4]
        else:
            raise ValueError(f"AFCPAdaptiveSelection expects 5- or 6-field batches, got {len(batch0)}")

        afcp = AFCPAdaptiveSelection(alpha=alpha, ttest_delta=getattr(exp_cfg, "afcp_ttest_delta", None), random_state=getattr(exp_cfg, "seed", 2024))
        C_sets_afcp, _k_hat = afcp.multiclass_classification(
            X_calib=X_calib,
            Y_calib=Y_calib,
            X_test=X_test,
            backbone=backbone,
            att_idx=att_idx,
            return_khat=True,
            conditional=False,
            left_tail=False,
            device=device,
            group_X_calib=group_X_calib,
            group_X_test=group_X_test,
            show_progress=True,
        )
        results["AFCP Adaptive Selection"] = prediction_sets_to_metrics(test_np, C_sets_afcp, alpha)

    hard_cp = calibrate_hard_cluster_cp(
        backbone,
        cal_loader,
        alpha=alpha,
        K=model_cfg.num_prototypes,
        device=device,
        seed=exp_cfg.hard_cluster_seed,
    )
    results["Hard Cluster CP"] = evaluate_hard_cluster_cp(backbone, hard_cp, test_loader, device=device, alpha=alpha)

    soft_assign = SoftPrototypeAssignment(
        backbone=backbone,
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
        alpha=alpha,
    )
    results["FaReG"] = evaluate_fareg_cp(
        backbone=backbone,
        cal_loader=cal_loader,
        test_loader=test_loader,
        alpha=alpha,
        device=device,
    )

    return results




from util.eval_tool import (
    evaluate_global_cp,
    evaluate_fixed_group_cp,
    calibrate_global_cp,
    calibrate_fixed_group_cp,
    calibrate_hard_cluster_cp,
    evaluate_hard_cluster_cp,
    calibrate_soft_prototype_cp,
    calibrate_sgcp,
    evaluate_soft_prototype_cp,
    evaluate_sg_cp,
)
from util.utils import loader_to_numpy
from util.train_tool import *

def evaluate_fareg_cp(backbone, cal_loader, test_loader, alpha, device="cpu"):
    dev = device if isinstance(device, torch.device) else torch.device(device)

    cal_np = loader_to_numpy(cal_loader)
    test_np = loader_to_numpy(test_loader)

    X_calib = cal_np["x"].astype(np.float32)
    Y_calib = cal_np["y"].astype(int)
    X_test = test_np["x"].astype(np.float32)

    class _BackboneProbNet:
        def __init__(self, backbone_model, device_):
            self.backbone = backbone_model
            self.device = device_

        def predict_prob(self, inputs):
            self.backbone.eval()
            with torch.no_grad():
                xb = inputs.to(self.device).float()
                logits, _ = self.backbone(xb)
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()

    class _Bbox:
        def __init__(self, net):
            self.net = net

    bbox_mc = _Bbox(_BackboneProbNet(backbone, dev))

    marginal = Marginal_Fairness(alpha=alpha, random_state=2025)
    C_sets_marginal_calib = marginal.multiclass_classification(
        X_calib,
        X_calib,
        Y_calib,
        bbox_mc=bbox_mc,
        left_tail=False,
        conditional=False,
    )
    covs_calib = np.array(
        [1.0 if int(Y_calib[i]) in set(C_sets_marginal_calib[i]) else 0.0 for i in range(len(Y_calib))],
        dtype=np.float32,
    )

    def _fareg_inputs(np_dict):
        color = np_dict["color"].astype(np.float32)
        age = (np_dict["age"].astype(np.float32) / 4.0) if np.max(np_dict["age"]) > 0 else np_dict["age"].astype(np.float32)
        region = (np_dict["region"].astype(np.float32) / 3.0) if np.max(np_dict["region"]) > 0 else np_dict["region"].astype(np.float32)
        return np.stack([color, age, region], axis=1).astype(np.float32)

    Xf_cal = _fareg_inputs(cal_np)
    Xf_test = _fareg_inputs(test_np)

    rng = np.random.default_rng(2025)
    perm = rng.permutation(len(Xf_cal))
    split = max(1, len(perm) // 2)
    tr_idx, va_idx = perm[:split], perm[split:]
    if len(va_idx) == 0:
        va_idx = tr_idx

    X_tr = torch.tensor(Xf_cal[tr_idx], dtype=torch.float32, device=dev)
    y_tr = torch.tensor(covs_calib[tr_idx], dtype=torch.float32, device=dev)
    X_va = torch.tensor(Xf_cal[va_idx], dtype=torch.float32, device=dev)
    y_va = torch.tensor(covs_calib[va_idx], dtype=torch.float32, device=dev)

    fareg = FaReGNet(num_features=Xf_cal.shape[1], delta=0.3, device=dev).to(dev)
    opt = torch.optim.Adam(fareg.parameters(), lr=1e-3)

    def _val_loss():
        fareg.eval()
        with torch.no_grad():
            out, _, _ = fareg(X_va)
            denom = torch.sum(out).clamp(min=1e-6)
            return float(torch.sum(out.squeeze() * y_va) / denom)

    best_state = None
    best_val = float("inf")

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=min(512, len(ds)), shuffle=True)
    for _epoch in range(200):
        fareg.train()
        for xb, covb in loader:
            opt.zero_grad()
            out, mu, logvar = fareg(xb)
            denom = torch.sum(out).clamp(min=1e-6)
            miscov_loss = torch.sum(out.squeeze() * covb) / denom
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = miscov_loss + 2.0 * kld
            loss.backward()
            opt.step()
        v = _val_loss()
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in fareg.state_dict().items()}

    if best_state is not None:
        fareg.load_state_dict(best_state)
    fareg.eval()

    with torch.no_grad():
        p_cal, _, _ = fareg(torch.tensor(Xf_cal, dtype=torch.float32, device=dev))
        p_test, _, _ = fareg(torch.tensor(Xf_test, dtype=torch.float32, device=dev))
    p_cal = p_cal.squeeze(-1).detach().cpu().numpy()
    p_test = p_test.squeeze(-1).detach().cpu().numpy()

    mask_list = []
    for _ in range(20):
        sel_cal = (rng.random(len(p_cal)) < p_cal).astype(int)
        sel_test = (rng.random(len(p_test)) < p_test).astype(int)
        mask_list.append((np.flatnonzero(sel_cal == 1).tolist(), np.flatnonzero(sel_test == 1).tolist()))

    fareg_method = FaReG_Fairness(alpha=alpha, random_state=2025)
    C_sets_fareg = fareg_method.multiclass_classification(
        X_calib,
        Y_calib,
        X_test,
        bbox_mc=bbox_mc,
        mask_list=mask_list,
        conditional=False,
        left_tail=False,
    )
    return prediction_sets_to_metrics(test_np, C_sets_fareg, alpha)


def evaluate_sgcp(backbone, train_loader, cal_loader, test_loader, exp_cfg, model_cfg, sgcp_cfg, device):
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
        backbone=backbone,
        epochs=sgcp_cfg.epochs,
        lr=sgcp_cfg.lr,
        beta_kl=sgcp_cfg.beta_kl,
        lambda_balance=sgcp_cfg.lambda_balance,
        lambda_score=sgcp_cfg.lambda_score,
        n_latent_samples=sgcp_cfg.train_latent_samples,
        num_score_bins=getattr(sgcp_cfg, "num_score_bins", 20),
        score_bin_edges=getattr(sgcp_cfg, "score_bin_edges", "quantile"),
        hist_smoothing=getattr(sgcp_cfg, "hist_smoothing", 1e-3),
        device=device,
    )
    sg_cp = calibrate_sgcp(
        backbone,
        sg_assign,
        cal_loader,
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
        device=device,
        n_latent_samples=sgcp_cfg.eval_latent_samples,
        alpha=alpha,
    )


    return results, sg_assign, sg_cp 
