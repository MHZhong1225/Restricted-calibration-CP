import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from SelectiveCI_fairness.methods import AFCPAdaptiveSelection
from SelectiveCI_fairness.sgcp_flow import StochasticAssignment, PrototypeAssignment

from FaReG.Group_fairness.cp import Marginal_Fairness, FaReG_Fairness
from FaReG.Group_fairness.networks import FaReG as FaReGNet

from util.eval_tool import (
    evaluate_global_cp,
    evaluate_fixed_group_cp,
    calibrate_global_cp,
    calibrate_fixed_group_cp,
    calibrate_hard_cluster_cp,
    evaluate_hard_cluster_cp,
    calibrate_prototype_cp,
    calibrate_sgcp,
    evaluate_prototype_cp,
    evaluate_sg_cp,
)
from util.utils import loader_to_numpy
from util.train_tool import train_prototype_assignment, train_stochastic_assignment

def prediction_sets_to_metrics(test_np, C_sets, alpha):
    y = test_np['y']
    attr1 = test_np['color']
    attr2 = test_np['age']
    attr3 = test_np['region']

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
    
    attr1_cov = grp(cover, attr1)
    attr2_cov = grp(cover, attr2)
    attr3_cov = grp(cover, attr3)
    
    attr4 = test_np.get("diag")
    attr4_cov = grp(cover, attr4) if attr4 is not None else {}
    attr4_size = grp(size, attr4) if attr4 is not None else {}
    
    attr5 = test_np.get("attr5")
    attr5_cov = grp(cover, attr5) if attr5 is not None else {}
    attr5_size = grp(size, attr5) if attr5 is not None else {}
    
    attr6 = test_np.get("attr6")
    attr6_cov = grp(cover, attr6) if attr6 is not None else {}
    attr6_size = grp(size, attr6) if attr6 is not None else {}
    
    attr7 = test_np.get("attr7")
    attr7_cov = grp(cover, attr7) if attr7 is not None else {}
    attr7_size = grp(size, attr7) if attr7 is not None else {}

    joint_keys = np.asarray([f"{int(c)}|{int(a)}|{int(r)}" for c, a, r in zip(attr1, attr2, attr3)], dtype=object)
    covgap_attr1 = cov_gap_from_coverages(attr1_cov)
    covgap_attr2 = cov_gap_from_coverages(attr2_cov)
    covgap_attr3 = cov_gap_from_coverages(attr3_cov)
    covgap_attr4 = cov_gap_from_coverages(attr4_cov) if attr4 is not None else float("nan")
    covgap = cov_gap_from_keys(joint_keys)
    
    out = {
        'overall_coverage': float(cover.mean()),
        'avg_set_size': float(size.mean()),
        'attr1_coverages': attr1_cov,
        'attr1_set_sizes': grp(size, attr1),
        'attr2_coverages': attr2_cov,
        'attr2_set_sizes': grp(size, attr2),
        'attr3_coverages': attr3_cov,
        'attr3_set_sizes': grp(size, attr3),
        'attr4_coverages': attr4_cov,
        'attr4_set_sizes': attr4_size,
        'blue_coverage': float(cover[attr1 == 1].mean()) if 1 in attr1 else float("nan"),
        'blue_avg_set_size': float(size[attr1 == 1].mean()) if 1 in attr1 else float("nan"),
        'covgap': covgap,
        'covgap_attr1': covgap_attr1,
        'covgap_attr2': covgap_attr2,
        'covgap_attr3': covgap_attr3,
        'covgap_attr4': covgap_attr4,
    }
    
    if attr4 is not None:
        attr4_attr1_gaps = {}
        for d in np.unique(attr4):
            idx_d = (attr4 == d)
            idx_c0 = idx_d & (attr1 == 0)
            idx_c1 = idx_d & (attr1 == 1)
            if idx_c0.any() and idx_c1.any():
                cov0 = float(cover[idx_c0].mean())
                cov1 = float(cover[idx_c1].mean())
                attr4_attr1_gaps[int(d)] = abs(cov0 - cov1)
        
        out["attr4_attr1_gaps"] = attr4_attr1_gaps
        gaps_list = list(attr4_attr1_gaps.values())
        out["fairgap_attr4_attr1_mean"] = float(np.mean(gaps_list)) if gaps_list else float("nan")
        out["fairgap_attr4_attr1_max"] = float(np.max(gaps_list)) if gaps_list else float("nan")
        
    if attr5 is not None:
        out.update({
            "attr5_coverages": attr5_cov, "attr5_set_sizes": attr5_size,
            "attr6_coverages": attr6_cov, "attr6_set_sizes": attr6_size,
            "attr7_coverages": attr7_cov, "attr7_set_sizes": attr7_size,
            "covgap_attr5": cov_gap_from_coverages(attr5_cov),
            "covgap_attr6": cov_gap_from_coverages(attr6_cov),
            "covgap_attr7": cov_gap_from_coverages(attr7_cov),
        })
        
    if 0 in attr1_cov and 1 in attr1_cov:
        out["fairgap_attr1"] = abs(attr1_cov[0] - attr1_cov[1])
    else:
        out["fairgap_attr1"] = float("nan")
        
    return out


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
        age_max = max(1.0, float(np.max(np_dict["age"])))
        age = np_dict["age"].astype(np.float32) / age_max
        reg_max = max(1.0, float(np.max(np_dict["region"])))
        region = np_dict["region"].astype(np.float32) / reg_max
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
        "Marginal CP": evaluate_global_cp(backbone, marginal_cp, test_loader, device=device, alpha=alpha),
        "Partial CP": evaluate_fixed_group_cp(
            backbone,
            partial_color_cp,
            test_loader,
            key_fn=lambda d: list(d["color"]),
            device=device,
            alpha=alpha,
        ),
        "Exhaustive CP": evaluate_fixed_group_cp(
            backbone,
            exhaustive_cp,
            test_loader,
            key_fn=lambda d: list(zip(d["color"], d["age"], d["region"])),
            device=device,
            alpha=alpha,
        ),
    }

    hard_cp = calibrate_hard_cluster_cp(
        backbone,
        cal_loader,
        alpha=alpha,
        K=model_cfg.num_prototypes,
        device=device,
        seed=exp_cfg.hard_cluster_seed,
    )
    results["Clustered CP"] = evaluate_hard_cluster_cp(backbone, hard_cp, test_loader, device=device, alpha=alpha)

    soft_assign = PrototypeAssignment(
        backbone=backbone,
        num_prototypes=model_cfg.num_prototypes,
        temperature=model_cfg.temperature,
    )
    soft_assign = train_prototype_assignment(
        soft_assign,
        train_loader,
        epochs=soft_cfg.epochs,
        lr=soft_cfg.lr,
        lambda_balance=soft_cfg.lambda_balance,
        device=device,
    )
    soft_cp = calibrate_prototype_cp(
        backbone,
        soft_assign,
        cal_loader,
        alpha=alpha,
        device=device,
        mode=soft_cfg.mode,
        gamma=soft_cfg.gamma,
    )
    results["Prototype CP"] = evaluate_prototype_cp(
        backbone,
        soft_assign,
        soft_cp,
        test_loader,
        dataset_name=exp_cfg.dataset_mode,
        device=device,
        alpha=alpha,
    )

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

        is_tabular = (X_calib.ndim == 2)

        if len(batch0) == 5:
            att_idx_emb = [d - 1, d - 2, d - 3]
            embedded_ok = False
            if is_tabular and n_check > 0:
                embedded_ok = float(np.mean(X_calib[:n_check, att_idx_emb[0]].astype(int) == cal_np["color"][:n_check].astype(int))) >= 0.95
            
            if embedded_ok:
                att_idx = att_idx_emb
            else:
                group_X_calib = np.column_stack([cal_np["color"], cal_np["age"], cal_np["region"]]).astype(int)
                group_X_test = np.column_stack([test_np["color"], test_np["age"], test_np["region"]]).astype(int)
                att_idx = [0, 1, 2]
                
        elif len(batch0) == 6:
            att_idx = [d - 1, d - 2, d - 4]
        elif len(batch0) == 7:
            group_X_calib = np.column_stack([cal_np["color"], cal_np["age"], cal_np["region"], cal_np["diag"]]).astype(int)
            group_X_test = np.column_stack([test_np["color"], test_np["age"], test_np["region"], test_np["diag"]]).astype(int)
            att_idx = [0, 1, 2, 3]
        elif len(batch0) == 9:
            group_X_calib = np.column_stack([cal_np["color"], cal_np["age"], cal_np["region"], cal_np["diag"], cal_np["attr5"], cal_np["attr6"], cal_np["attr7"]]).astype(int)
            group_X_test = np.column_stack([test_np["color"], test_np["age"], test_np["region"], test_np["diag"], test_np["attr5"], test_np["attr6"], test_np["attr7"]]).astype(int)
            att_idx = [0, 1, 2, 3, 4, 5, 6]
        else:
            raise ValueError(f"AFCPAdaptiveSelection expects 5-, 6-, 7-, or 9-field batches, got {len(batch0)}")

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
        results["AFCP"] = prediction_sets_to_metrics(test_np, C_sets_afcp, alpha)

    results["FaReG"] = evaluate_fareg_cp(
        backbone=backbone,
        cal_loader=cal_loader,
        test_loader=test_loader,
        alpha=alpha,
        device=device,
    )

    return results

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
        dataset_name = exp_cfg.dataset_mode,
        device=device,
        n_latent_samples=sgcp_cfg.eval_latent_samples,
        alpha=alpha,
    )

    return results, sg_assign, sg_cp
