import numpy as np
import torch
from .utils import conformal_quantile, extract_all
from SelectiveCI_fairness.cp_obj import *
from util.utils import prediction_set_from_probs_and_thresholds
from util.utils import simple_kmeans, extract_sgcp_weights, extract_proto_weights, weighted_quantile

# =========================
# Calibration
# =========================

@torch.no_grad()
def calibrate_global_cp(backbone, cal_loader, alpha=0.1, device="cpu"):
    data = extract_all(backbone, cal_loader, device=device)
    return GlobalCP(threshold=conformal_quantile(data["scores"], alpha))


def evaluate_prediction_sets(pred_sets, y_true, attr1, attr2, attr3, attr4=None, attr5=None, attr6=None, attr7=None, alpha=None):
    cover = np.array([int(y_true[i] in pred_sets[i]) for i in range(len(y_true))])
    sizes = np.array([len(s) for s in pred_sets])

    def group_stats(keys):
        covs, szs = {}, {}
        for g in np.unique(keys):
            idx = keys == g
            covs[int(g)] = float(cover[idx].mean())
            szs[int(g)] = float(sizes[idx].mean())
        return covs, szs

    attr1_covs, attr1_sizes = group_stats(attr1)
    attr2_covs, attr2_sizes = group_stats(attr2)
    attr3_covs, attr3_sizes = group_stats(attr3)
    if attr4 is not None:
        attr4_covs, attr4_sizes = group_stats(attr4)
    else:
        attr4_covs, attr4_sizes = {}, {}
        
    attr5_covs, attr5_sizes = group_stats(attr5) if attr5 is not None else ({}, {})
    attr6_covs, attr6_sizes = group_stats(attr6) if attr6 is not None else ({}, {})
    attr7_covs, attr7_sizes = group_stats(attr7) if attr7 is not None else ({}, {})

    out = {
        "overall_coverage": float(cover.mean()),
        "avg_set_size": float(sizes.mean()),
        "attr1_coverages": attr1_covs,
        "attr1_set_sizes": attr1_sizes,
        "attr2_coverages": attr2_covs,
        "attr2_set_sizes": attr2_sizes,
        "attr3_coverages": attr3_covs,
        "attr3_set_sizes": attr3_sizes,
        "attr4_coverages": attr4_covs,
        "attr4_set_sizes": attr4_sizes,
    }
    
    # Only calculate blue metrics if 1 exists in attr1 and there are exactly 2 unique values (like color)
    if 1 in attr1 and len(np.unique(attr1)) <= 2:
        out["blue_coverage"] = float(cover[attr1 == 1].mean())
        out["blue_avg_set_size"] = float(sizes[attr1 == 1].mean())
    else:
        out["blue_coverage"] = float("nan")
        out["blue_avg_set_size"] = float("nan")

    if attr5 is not None:
        out.update({
            "attr5_coverages": attr5_covs, "attr5_set_sizes": attr5_sizes,
            "attr6_coverages": attr6_covs, "attr6_set_sizes": attr6_sizes,
            "attr7_coverages": attr7_covs, "attr7_set_sizes": attr7_sizes,
        })

    if alpha is not None:
        target = 1.0 - float(alpha)
        def cov_gap_from_coverages(coverages):
            vals = [v for v in coverages.values() if not np.isnan(v)]
            if len(vals) == 0:
                return float("nan")
            return float(np.mean(np.abs(np.asarray(vals, dtype=float) - target)))
        def cov_gap_from_keys(keys):
            uniq = np.unique(keys)
            if len(uniq) == 0:
                return float("nan")
            gaps = []
            for k in uniq:
                idx = keys == k
                if not idx.any():
                    continue
                gaps.append(abs(float(cover[idx].mean()) - target))
            if len(gaps) == 0:
                return float("nan")
            return float(np.mean(np.asarray(gaps, dtype=float)))
        
        joint_keys = np.asarray([f"{int(c)}|{int(a)}|{int(r)}" for c, a, r in zip(attr1, attr2, attr3)], dtype=object)
        out["covgap"] = cov_gap_from_keys(joint_keys)
        out["covgap_attr1"] = cov_gap_from_coverages(attr1_covs)
        out["covgap_attr2"] = cov_gap_from_coverages(attr2_covs)
        out["covgap_attr3"] = cov_gap_from_coverages(attr3_covs)
        
        if attr4 is not None:
            out["covgap_attr4"] = cov_gap_from_coverages(attr4_covs)
            
            # Additional fairness gap for attr4 (minority consistency)
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
            out["covgap_attr5"] = cov_gap_from_coverages(attr5_covs)
            out["covgap_attr6"] = cov_gap_from_coverages(attr6_covs)
            out["covgap_attr7"] = cov_gap_from_coverages(attr7_covs)
            
    # Overall attr1 fairness gap (max difference across all pairs if >1 groups)
    if len(attr1_covs) >= 2:
        vals = list(attr1_covs.values())
        out["fairgap_attr1"] = float(np.max(vals) - np.min(vals))
    else:
        out["fairgap_attr1"] = float("nan")

    return out


@torch.no_grad()
def calibrate_fixed_group_cp(backbone, cal_loader, key_fn, alpha=0.1, min_group_n=8, device="cpu"):
    data = extract_all(backbone, cal_loader, device=device)
    scores = data["scores"]
    global_q = conformal_quantile(scores, alpha)

    keys = key_fn(data)
    uniq = sorted(set(keys))

    thresholds = {}
    for k in uniq:
        idx = np.where(np.array(keys, dtype=object) == k)[0]
        thresholds[k] = global_q if len(idx) < min_group_n else conformal_quantile(scores[idx], alpha)

    return FixedGroupCP(thresholds=thresholds, fallback_threshold=global_q)


@torch.no_grad()
def calibrate_hard_cluster_cp(backbone, cal_loader, alpha=0.1, K=8, device="cpu", seed=42):
    data = extract_all(backbone, cal_loader, device=device)
    feats, scores = data["feats"], data["scores"]

    centers = simple_kmeans(feats, K=K, seed=seed)
    dists = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    clusters = np.argmin(dists, axis=1)

    global_q = conformal_quantile(scores, alpha)
    thresholds = np.zeros(K, dtype=np.float32)

    for k in range(K):
        idx = np.where(clusters == k)[0]
        thresholds[k] = global_q if len(idx) < 8 else conformal_quantile(scores[idx], alpha)

    return ClusteredCP(cluster_thresholds=thresholds, cluster_centers=centers)


@torch.no_grad()
def calibrate_prototype_cp(backbone, assign_model, cal_loader, alpha=0.1, device="cpu", mode="top1", gamma=1.0):
    data = extract_all(backbone, cal_loader, device=device)
    scores = data["scores"]
    weights = extract_proto_weights(assign_model, cal_loader, device=device)
    K = weights.shape[1]

    global_q = conformal_quantile(scores, alpha)
    thresholds = np.zeros(K, dtype=np.float32)

    for k in range(K):
        w = weights[:, k]
        thresholds[k] = global_q if np.sum(w > 1e-4) < 8 else weighted_quantile(scores, w, 1 - alpha)

    return PrototypeCP(prototype_thresholds=thresholds, mode=mode, gamma=gamma)


@torch.no_grad()
def calibrate_sgcp(backbone, assign_model, cal_loader, alpha=0.1, device="cpu", n_latent_samples=10):
    data = extract_all(backbone, cal_loader, device=device)
    scores = np.asarray(data["scores"], dtype=np.float32)
    weights = extract_sgcp_weights(assign_model, cal_loader, device=device, n_latent_samples=n_latent_samples).astype(np.float32)

    row_sum = np.sum(weights, axis=1, keepdims=True)
    row_sum = np.clip(row_sum, 1e-12, None)
    weights = weights / row_sum

    def build_weighted_ecdf(s, w):
        s = np.asarray(s, dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)
        order = np.argsort(s)
        s_sorted = s[order]
        w_sorted = w[order]
        denom = float(np.sum(w_sorted))
        if denom <= 0.0:
            return s_sorted, np.zeros_like(s_sorted, dtype=np.float32)
        cdf = (np.cumsum(w_sorted) / denom).astype(np.float32)
        return s_sorted, cdf

    global_scores_sorted, global_cdf_sorted = build_weighted_ecdf(scores, np.ones_like(scores, dtype=np.float32))

    k = weights.shape[1]
    group_scores_sorted = []
    group_cdf_sorted = []
    for g in range(k):
        w_g = weights[:, g]
        if float(np.sum(w_g)) <= 1e-8:
            group_scores_sorted.append(global_scores_sorted)
            group_cdf_sorted.append(global_cdf_sorted)
        else:
            s_sorted, cdf_sorted = build_weighted_ecdf(scores, w_g)
            group_scores_sorted.append(s_sorted)
            group_cdf_sorted.append(cdf_sorted)

    sgcp = SGCP(group_scores_sorted=group_scores_sorted, group_cdf_sorted=group_cdf_sorted, q_v=0.0)
    v_cal = sgcp.local_cdf(scores, weights)
    sgcp.q_v = conformal_quantile(v_cal, alpha)
    return sgcp



@torch.no_grad()
def evaluate_global_cp(backbone, cp_obj, test_loader, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = np.full(len(data["y"]), cp_obj.threshold, dtype=np.float32)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
        
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], 
                                    attr4=data.get("diag"), attr5=data.get("attr5"), 
                                    attr6=data.get("attr6"), attr7=data.get("attr7"), alpha=alpha)


@torch.no_grad()
def evaluate_fixed_group_cp(backbone, cp_obj, test_loader, key_fn, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    keys = key_fn(data)
    thresholds = cp_obj.threshold_for_keys(keys)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
        
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], 
                                    attr4=data.get("diag"), attr5=data.get("attr5"), 
                                    attr6=data.get("attr6"), attr7=data.get("attr7"), alpha=alpha)


@torch.no_grad()
def evaluate_hard_cluster_cp(backbone, cp_obj, test_loader, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(data["feats"])
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
        
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], 
                                    attr4=data.get("diag"), attr5=data.get("attr5"), 
                                    attr6=data.get("attr6"), attr7=data.get("attr7"), alpha=alpha)


@torch.no_grad()
def evaluate_prototype_cp(backbone, assign_model, cp_obj, test_loader, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    weights = extract_proto_weights(assign_model, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(weights)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
        
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], 
                                    attr4=data.get("diag"), attr5=data.get("attr5"), 
                                    attr6=data.get("attr6"), attr7=data.get("attr7"), alpha=alpha)


@torch.no_grad()
def evaluate_sg_cp(backbone, assign_model, cp_obj, test_loader, device="cpu", n_latent_samples=10, alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    probs = np.asarray(data["probs"], dtype=np.float32)
    weights = extract_sgcp_weights(assign_model, test_loader, device=device, n_latent_samples=n_latent_samples).astype(np.float32)
    row_sum = np.sum(weights, axis=1, keepdims=True)
    row_sum = np.clip(row_sum, 1e-12, None)
    weights = weights / row_sum

    scores_all = 1.0 - probs
    v_all = cp_obj.local_cdf(scores_all, weights)
    pred_sets = []
    n, c = probs.shape
    qv = float(cp_obj.q_v)
    for i in range(n):
        labels = [y for y in range(c) if float(v_all[i, y]) <= qv]
        pred_sets.append(labels)

    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], 
                                    attr4=data.get("diag"), attr5=data.get("attr5"), 
                                    attr6=data.get("attr6"), attr7=data.get("attr7"), alpha=alpha)