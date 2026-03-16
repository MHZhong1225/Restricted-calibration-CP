import numpy as np
import torch
from .utils import conformal_quantile, extract_all
from SelectiveCI_fairness.cp_obj import *
from util.utils import extract_all, conformal_quantile, prediction_set_from_probs_and_thresholds
from util.utils import simple_kmeans, extract_sls_difficulties, extract_softproto_weights, weighted_quantile

# =========================
# Calibration
# =========================


@torch.no_grad()
def calibrate_global_cp(backbone, cal_loader, alpha=0.1, device="cpu"):
    data = extract_all(backbone, cal_loader, device=device)
    return GlobalCP(threshold=conformal_quantile(data["scores"], alpha))



def evaluate_prediction_sets(pred_sets, y_true, color, age, region):
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

    return {
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
def calibrate_sls_cp(backbone, assign_model, cal_loader, alpha=0.1, device="cpu", n_latent_samples=10, num_bins=5, min_bin_n=30):
    data = extract_all(backbone, cal_loader, device=device)
    scores = data["scores"]
    difficulties = extract_sls_difficulties(assign_model, cal_loader, device=device, n_latent_samples=n_latent_samples)
    difficulties = np.clip(difficulties, 0.0, 1.0)

    global_q = conformal_quantile(scores, alpha)

    raw_edges = np.quantile(difficulties, np.linspace(0.0, 1.0, num_bins + 1))
    raw_edges[0] = 0.0
    raw_edges[-1] = 1.0
    bin_edges = raw_edges.copy()
    for i in range(1, len(bin_edges)):
        bin_edges[i] = max(bin_edges[i], bin_edges[i - 1])

    bin_ids = np.digitize(difficulties, bin_edges[1:-1], right=False)
    bin_thresholds = np.full(num_bins, global_q, dtype=np.float32)

    for b in range(num_bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) >= min_bin_n:
            bin_thresholds[b] = conformal_quantile(scores[idx], alpha)

    return BinnedSLSCP(
        bin_edges=bin_edges.astype(np.float32),
        bin_thresholds=bin_thresholds,
        fallback_threshold=global_q,
    )



@torch.no_grad()
def evaluate_global_cp(backbone, cp_obj, test_loader, device="cpu"):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = np.full(len(data["y"]), cp_obj.threshold, dtype=np.float32)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"])


@torch.no_grad()
def evaluate_fixed_group_cp(backbone, cp_obj, test_loader, key_fn, device="cpu"):
    data = extract_all(backbone, test_loader, device=device)
    keys = key_fn(data)
    thresholds = cp_obj.threshold_for_keys(keys)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"])


@torch.no_grad()
def evaluate_hard_cluster_cp(backbone, cp_obj, test_loader, device="cpu"):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(data["feats"])
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"])


@torch.no_grad()
def evaluate_soft_prototype_cp(backbone, assign_model, cp_obj, test_loader, device="cpu"):
    data = extract_all(backbone, test_loader, device=device)
    weights = extract_softproto_weights(assign_model, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(weights)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"])


@torch.no_grad()
def evaluate_sls_cp(backbone, assign_model, cp_obj, test_loader, device="cpu", n_latent_samples=10):
    data = extract_all(backbone, test_loader, device=device)
    difficulties = extract_sls_difficulties(assign_model, test_loader, device=device, n_latent_samples=n_latent_samples)
    thresholds = cp_obj.threshold_for_batch(difficulties)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)
    return evaluate_prediction_sets(pred_sets, data["y"], data["color"], data["age"], data["region"])
