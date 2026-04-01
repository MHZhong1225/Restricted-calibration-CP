import numpy as np
import torch

from .utils import conformal_quantile, extract_all
from SelectiveCI_fairness.cp_obj import *
from util.utils import (
    prediction_set_from_probs_and_thresholds,
    simple_kmeans,
    extract_sgcp_weights,
    extract_proto_weights,
    weighted_quantile,
)

# =========================================================
# Dataset-specific attribute specification
# =========================================================
# Unified internal interface:
#   data["attr1"], data["attr2"], ..., data["attr7"]
#
# Dataset meanings:
#   mimic:
#       attr1 = minority
#       attr2 = gender_m
#       attr3 = public_insurance
#
#   nursery:
#       attr1 = parents
#       attr2 = has_nurs
#       attr3 = form
#       attr4 = children
#       attr5 = finance
#       attr6 = social
#       attr7 = health
#
#   bach:
#       attr1 = y  (true class as group)
#
#   syn:
#       attr1 = Color
#       attr2 = Age Group
#       attr3 = Region
#
DATASET_ATTR_SPECS = {
    "mimic": {
        "attr_names": ["attr1", "attr2", "attr3"],
        "attr_meanings": {
            "attr1": "minority",
            "attr2": "gender_m",
            "attr3": "public_insurance",
        },
        "primary_attr": "attr1",
        "compute_blue_metrics": False,
    },
    "nursery": {
        "attr_names": ["attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "attr7"],
        "attr_meanings": {
            "attr1": "parents",
            "attr2": "has_nurs",
            "attr3": "form",
            "attr4": "children",
            "attr5": "finance",
            "attr6": "social",
            "attr7": "health",
        },
        "primary_attr": "attr1",
        "compute_blue_metrics": False,
    },
    "bach": {
        "attr_names": ["attr1"],
        "attr_meanings": {
            "attr1": "label_group",
        },
        "primary_attr": "attr1",
        "compute_blue_metrics": False,
    },
    "syn": {
        "attr_names": ["attr1", "attr2", "attr3"],
        "attr_meanings": {
            "attr1": "color",
            "attr2": "age_group",
            "attr3": "region",
        },
        "primary_attr": "attr1",
        "compute_blue_metrics": True,
    },
}


# =========================================================
# Helpers
# =========================================================
def get_dataset_attr_spec(dataset_name: str):
    if dataset_name not in DATASET_ATTR_SPECS:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}. "
            f"Supported datasets: {list(DATASET_ATTR_SPECS.keys())}"
        )
    return DATASET_ATTR_SPECS[dataset_name]


def get_eval_attrs(data: dict, dataset_name: str):
    """
    Extract only the attributes relevant for the given dataset.

    Returns
    -------
    attrs : dict
        e.g. {"attr1": ..., "attr2": ..., ...}
    primary_attr : str
    compute_blue_metrics : bool
    attr_meanings : dict
        e.g. {"attr1": "minority", ...}
    """
    spec = get_dataset_attr_spec(dataset_name)
    attrs = {k: data.get(k) for k in spec["attr_names"] if data.get(k) is not None}
    return attrs, spec["primary_attr"], spec["compute_blue_metrics"], spec["attr_meanings"]


def _safe_to_numpy(x):
    if x is None:
        return None
    return np.asarray(x)


def _group_stats(values, cover, sizes):
    values = _safe_to_numpy(values)
    covs, szs = {}, {}

    if values is None:
        return covs, szs

    uniq = np.unique(values)
    for g in uniq:
        idx = values == g
        if idx.sum() == 0:
            continue
        try:
            key = int(g)
        except Exception:
            key = str(g)
        covs[key] = float(cover[idx].mean())
        szs[key] = float(sizes[idx].mean())
    return covs, szs


def _fairness_gap_from_coverages(coverages):
    vals = [v for v in coverages.values() if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    vals = np.asarray(vals, dtype=float)
    return float(np.max(vals) - np.min(vals))


def _cov_gap_from_coverages(coverages, target):
    vals = [v for v in coverages.values() if not np.isnan(v)]
    if len(vals) == 0:
        return float("nan")
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(np.abs(vals - target)))


def _cov_gap_from_keys(keys, cover, target):
    keys = np.asarray(keys, dtype=object)
    uniq = np.unique(keys)
    if len(uniq) == 0:
        return float("nan")

    gaps = []
    for k in uniq:
        idx = keys == k
        if idx.sum() == 0:
            continue
        gaps.append(abs(float(cover[idx].mean()) - target))

    if len(gaps) == 0:
        return float("nan")
    return float(np.mean(np.asarray(gaps, dtype=float)))


def _build_joint_keys(attrs: dict, n: int):
    used_names = [k for k, v in attrs.items() if v is not None]
    if len(used_names) == 0:
        return None

    joint_keys = []
    for i in range(n):
        parts = []
        for name in used_names:
            v = attrs[name][i]
            try:
                parts.append(str(int(v)))
            except Exception:
                parts.append(str(v))
        joint_keys.append("|".join(parts))
    return np.asarray(joint_keys, dtype=object)


# =========================================================
# Calibration
# =========================================================
@torch.no_grad()
def calibrate_global_cp(backbone, cal_loader, alpha=0.1, device="cpu"):
    data = extract_all(backbone, cal_loader, device=device)
    return GlobalCP(threshold=conformal_quantile(data["scores"], alpha))


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
    weights = extract_sgcp_weights(
        assign_model, cal_loader, device=device, n_latent_samples=n_latent_samples
    ).astype(np.float32)

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

    global_scores_sorted, global_cdf_sorted = build_weighted_ecdf(
        scores, np.ones_like(scores, dtype=np.float32)
    )

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

    sgcp = SGCP(
        group_scores_sorted=group_scores_sorted,
        group_cdf_sorted=group_cdf_sorted,
        q_v=0.0,
    )
    v_cal = sgcp.local_cdf(scores, weights)
    sgcp.q_v = conformal_quantile(v_cal, alpha)
    return sgcp


# =========================================================
# Evaluation
# =========================================================
def evaluate_prediction_sets(
    pred_sets,
    y_true,
    attrs: dict,
    alpha=None,
    primary_attr=None,
    compute_blue_metrics=False,
    attr_meanings=None,
):
    """
    Unified evaluation for prediction sets.

    Parameters
    ----------
    pred_sets : list[list[int]]
    y_true : array-like
    attrs : dict
        e.g. {"attr1": ..., "attr2": ..., ...}
    alpha : float or None
    primary_attr : str or None
    compute_blue_metrics : bool
        Only True for syn, where attr1 == Blue/Grey and group 1 means Blue.
    attr_meanings : dict or None
        Human-readable names, e.g. {"attr1": "minority", ...}
    """
    y_true = np.asarray(y_true)
    cover = np.array([int(y_true[i] in pred_sets[i]) for i in range(len(y_true))], dtype=float)
    sizes = np.array([len(s) for s in pred_sets], dtype=float)

    out = {
        "overall_coverage": float(cover.mean()),
        "avg_set_size": float(sizes.mean()),
    }

    if attr_meanings is not None:
        out["attr_meanings"] = dict(attr_meanings)

    attr_coverages = {}

    # per-attribute metrics
    for attr_name, attr_vals in attrs.items():
        if attr_vals is None:
            continue

        covs, szs = _group_stats(attr_vals, cover, sizes)
        out[f"{attr_name}_coverages"] = covs
        out[f"{attr_name}_set_sizes"] = szs
        out[f"fairgap_{attr_name}"] = _fairness_gap_from_coverages(covs)
        attr_coverages[attr_name] = covs

    # primary fairness gap
    if primary_attr is not None and f"{primary_attr}_coverages" in out:
        out["fairgap_primary"] = _fairness_gap_from_coverages(out[f"{primary_attr}_coverages"])
    else:
        out["fairgap_primary"] = float("nan")

    # optional syn-only blue metrics
    if compute_blue_metrics and primary_attr in attrs and attrs[primary_attr] is not None:
        primary_vals = np.asarray(attrs[primary_attr])
        uniq = np.unique(primary_vals)
        if 1 in uniq and len(uniq) <= 2:
            idx_blue = primary_vals == 1
            out["blue_coverage"] = float(cover[idx_blue].mean())
            out["blue_avg_set_size"] = float(sizes[idx_blue].mean())
        else:
            out["blue_coverage"] = float("nan")
            out["blue_avg_set_size"] = float("nan")
    else:
        out["blue_coverage"] = float("nan")
        out["blue_avg_set_size"] = float("nan")

    # alpha-based covgap
    if alpha is not None:
        target = 1.0 - float(alpha)

        for attr_name, covs in attr_coverages.items():
            out[f"covgap_{attr_name}"] = _cov_gap_from_coverages(covs, target)

        joint_keys = _build_joint_keys(attrs, len(y_true))
        out["covgap"] = (
            _cov_gap_from_keys(joint_keys, cover, target)
            if joint_keys is not None else float("nan")
        )

    return out


@torch.no_grad()
def evaluate_global_cp(backbone, cp_obj, test_loader, dataset_name, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = np.full(len(data["y"]), cp_obj.threshold, dtype=np.float32)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)

    attrs, primary_attr, compute_blue_metrics, attr_meanings = get_eval_attrs(data, dataset_name)

    return evaluate_prediction_sets(
        pred_sets=pred_sets,
        y_true=data["y"],
        attrs=attrs,
        alpha=alpha,
        primary_attr=primary_attr,
        compute_blue_metrics=compute_blue_metrics,
        attr_meanings=attr_meanings,
    )


@torch.no_grad()
def evaluate_fixed_group_cp(backbone, cp_obj, test_loader, key_fn, dataset_name, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    keys = key_fn(data)
    thresholds = cp_obj.threshold_for_keys(keys)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)

    attrs, primary_attr, compute_blue_metrics, attr_meanings = get_eval_attrs(data, dataset_name)

    return evaluate_prediction_sets(
        pred_sets=pred_sets,
        y_true=data["y"],
        attrs=attrs,
        alpha=alpha,
        primary_attr=primary_attr,
        compute_blue_metrics=compute_blue_metrics,
        attr_meanings=attr_meanings,
    )


@torch.no_grad()
def evaluate_hard_cluster_cp(backbone, cp_obj, test_loader, dataset_name, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(data["feats"])
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)

    attrs, primary_attr, compute_blue_metrics, attr_meanings = get_eval_attrs(data, dataset_name)

    return evaluate_prediction_sets(
        pred_sets=pred_sets,
        y_true=data["y"],
        attrs=attrs,
        alpha=alpha,
        primary_attr=primary_attr,
        compute_blue_metrics=compute_blue_metrics,
        attr_meanings=attr_meanings,
    )


@torch.no_grad()
def evaluate_prototype_cp(backbone, assign_model, cp_obj, test_loader, dataset_name, device="cpu", alpha=None):
    data = extract_all(backbone, test_loader, device=device)
    weights = extract_proto_weights(assign_model, test_loader, device=device)
    thresholds = cp_obj.threshold_for_batch(weights)
    pred_sets = prediction_set_from_probs_and_thresholds(data["probs"], thresholds)

    attrs, primary_attr, compute_blue_metrics, attr_meanings = get_eval_attrs(data, dataset_name)

    return evaluate_prediction_sets(
        pred_sets=pred_sets,
        y_true=data["y"],
        attrs=attrs,
        alpha=alpha,
        primary_attr=primary_attr,
        compute_blue_metrics=compute_blue_metrics,
        attr_meanings=attr_meanings,
    )


@torch.no_grad()
def evaluate_sg_cp(
    backbone,
    assign_model,
    cp_obj,
    test_loader,
    dataset_name,
    device="cpu",
    n_latent_samples=10,
    alpha=None,
):
    data = extract_all(backbone, test_loader, device=device)
    probs = np.asarray(data["probs"], dtype=np.float32)
    weights = extract_sgcp_weights(
        assign_model, test_loader, device=device, n_latent_samples=n_latent_samples
    ).astype(np.float32)

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

    attrs, primary_attr, compute_blue_metrics, attr_meanings = get_eval_attrs(data, dataset_name)

    return evaluate_prediction_sets(
        pred_sets=pred_sets,
        y_true=data["y"],
        attrs=attrs,
        alpha=alpha,
        primary_attr=primary_attr,
        compute_blue_metrics=compute_blue_metrics,
        attr_meanings=attr_meanings,
    )