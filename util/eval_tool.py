import numpy as np
import torch
from .utils import conformal_quantile, extract_all
from SelectiveCI_fairness.cp_obj import *
from util.utils import prediction_set_from_probs_and_thresholds
from util.utils import simple_kmeans, extract_sgcp_weights, extract_softproto_weights, weighted_quantile

# =========================
# Calibration
# =========================


@torch.no_grad()
def calibrate_global_cp(backbone, cal_loader, alpha=0.1, device="cpu"):
    data = extract_all(backbone, cal_loader, device=device)
    return GlobalCP(threshold=conformal_quantile(data["scores"], alpha))



def evaluate_prediction_sets(pred_sets, y_true, color, age, region, alpha=None):
    cover = np.array([int(y_true[i] in pred_sets[i]) for i in range(len(y_true))])
    sizes = np.array([len(s) for s in pred_sets])

    def group_stats(keys):
        covs, szs = {}, {}
        for g in np.unique(keys):
            idx = keys == g
            covs[int(g)] = float(cover[idx].mean())
            szs[int(g)] = float(sizes[idx].mean())
        return covs, szs

    color_covs, color_sizes = group_stats(color)
    age_covs, age_sizes = group_stats(age)
    region_covs, region_sizes = group_stats(region)

    out = {
        "overall_coverage": float(cover.mean()),
        "avg_set_size": float(sizes.mean()),
        "color_coverages": color_covs,
        "color_set_sizes": color_sizes,
        "age_coverages": age_covs,
        "age_set_sizes": age_sizes,
        "region_coverages": region_covs,
        "region_set_sizes": region_sizes,
        "blue_coverage": float(cover[color == 1].mean()),
        "blue_avg_set_size": float(sizes[color == 1].mean()),
    }
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
        joint_keys = np.asarray([f"{int(c)}|{int(a)}|{int(r)}" for c, a, r in zip(color, age, region)], dtype=object)
        out["covgap"] = cov_gap_from_keys(joint_keys)
        out["covgap_color"] = cov_gap_from_coverages(color_covs)
        out["covgap_age"] = cov_gap_from_coverages(age_covs)
        out["covgap_region"] = cov_gap_from_coverages(region_covs)
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

    return HardClusterCP(cluster_thresholds=thresholds, cluster_centers=centers)


@torch.no_grad()
def calibrate_soft_prototype_cp(backbone, assign_model, cal_loader, alpha=0.1, device="cpu", mode="top1", gamma=1.0):
    data = extract_all(backbone, cal_loader, device=device)
    scores = data["scores"]
    weights = extract_softproto_weights(assign_model, cal_loader, device=device)
    K = weights.shape[1]

    global_q = conformal_quantile(scores, alpha)
    thresholds = np.zeros(K, dtype=np.float32)

    for k in range(K):
        w = weights[:, k]
        thresholds[k] = global_q if np.sum(w > 1e-4) < 8 else weighted_quantile(scores, w, 1 - alpha)

    return SoftPrototypeCP(prototype_thresholds=thresholds, mode=mode, gamma=gamma)


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
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], alpha=alpha)


@torch.no_grad()
def evaluate_fixed_group_cp(backbone, cp_obj, test_loader, key_fn, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    keys = key_fn(data)
    thresholds = cp_obj.threshold_for_keys(keys)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], alpha=alpha)


@torch.no_grad()
def evaluate_hard_cluster_cp(backbone, cp_obj, test_loader, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(data["feats"])
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], alpha=alpha)


@torch.no_grad()
def evaluate_soft_prototype_cp(backbone, assign_model, cp_obj, test_loader, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    weights = extract_softproto_weights(assign_model, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(weights)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], alpha=alpha)


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
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"], alpha=alpha)
