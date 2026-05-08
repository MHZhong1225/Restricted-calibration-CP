import numpy as np
import torch

from CP_fairness.cp_obj import SGCP
from util.utils import conformal_quantile, extract_all, extract_sgcp_weights


DATASET_ATTR_SPECS = {
    "syn": {
        "attr_names": ["attr1", "attr2", "attr3"],
        "attr_meanings": {
            "attr1": "gender",
            "attr2": "phenotype",
            "attr3": "age",
        },
        "primary_attr": "attr1",
    },
    "mimic": {
        "attr_names": ["attr1", "attr2", "attr3"],
        "attr_meanings": {
            "attr1": "minority",
            "attr2": "gender_m",
            "attr3": "public_insurance",
        },
        "primary_attr": "attr1",
    },
    "nursery": {
        "attr_names": ["attr1", "attr2", "attr3", "attr4", "attr5"],
        "attr_meanings": {
            "attr1": "parents",
            "attr2": "children",
            "attr3": "finance",
            "attr4": "social",
            "attr5": "health",
        },
        "primary_attr": "attr1",
    },
    "bach": {
        "attr_names": ["attr1"],
        "attr_meanings": {"attr1": "label_group"},
        "primary_attr": "attr1",
    },
}


def get_eval_attrs(data: dict, dataset_name: str):
    spec = DATASET_ATTR_SPECS.get(dataset_name)
    if spec is None:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}. Supported: {list(DATASET_ATTR_SPECS)}")
    attrs = {k: data.get(k) for k in spec["attr_names"] if data.get(k) is not None}
    return attrs, spec["primary_attr"], spec["attr_meanings"]


def _group_stats(values, cover, sizes):
    values = np.asarray(values)
    covs, szs = {}, {}
    for g in np.unique(values):
        idx = values == g
        key = int(g) if np.issubdtype(np.asarray([g]).dtype, np.integer) else str(g)
        covs[key] = float(cover[idx].mean())
        szs[key] = float(sizes[idx].mean())
    return covs, szs


def _fairness_gap(coverages):
    vals = [v for v in coverages.values() if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    vals = np.asarray(vals, dtype=float)
    return float(np.max(vals) - np.min(vals))


def _coverage_gap(coverages, target):
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


def _joint_keys(attrs: dict, n: int):
    names = [name for name, values in attrs.items() if values is not None]
    if len(names) == 0:
        return None
    keys = []
    for i in range(n):
        parts = []
        for name in names:
            v = attrs[name][i]
            try:
                parts.append(str(int(v)))
            except Exception:
                parts.append(str(v))
        keys.append("|".join(parts))
    return np.asarray(keys, dtype=object)


def evaluate_prediction_sets(pred_sets, y_true, attrs: dict, alpha=None, primary_attr=None, attr_meanings=None):
    y_true = np.asarray(y_true)
    cover = np.asarray([int(y_true[i] in pred_sets[i]) for i in range(len(y_true))], dtype=float)
    sizes = np.asarray([len(s) for s in pred_sets], dtype=float)

    metrics = {
        "overall_coverage": float(cover.mean()),
        "avg_set_size": float(sizes.mean()),
    }
    if attr_meanings is not None:
        metrics["attr_meanings"] = dict(attr_meanings)

    attr_coverages = {}
    for attr_name, attr_vals in attrs.items():
        covs, szs = _group_stats(attr_vals, cover, sizes)
        metrics[f"{attr_name}_coverages"] = covs
        metrics[f"{attr_name}_set_sizes"] = szs
        metrics[f"fairgap_{attr_name}"] = _fairness_gap(covs)
        attr_coverages[attr_name] = covs

    metrics["fairgap_primary"] = (
        _fairness_gap(metrics[f"{primary_attr}_coverages"])
        if primary_attr is not None and f"{primary_attr}_coverages" in metrics
        else float("nan")
    )

    if alpha is not None:
        target = 1.0 - float(alpha)
        for attr_name, covs in attr_coverages.items():
            metrics[f"covgap_{attr_name}"] = _coverage_gap(covs, target)

        joint = _joint_keys(attrs, len(y_true))
        metrics["covgap"] = _cov_gap_from_keys(joint, cover, target) if joint is not None else float("nan")

    return metrics


def _normalize_rows(weights):
    row_sum = np.sum(weights, axis=1, keepdims=True)
    row_sum = np.clip(row_sum, 1e-12, None)
    return weights / row_sum


def _weighted_ecdf(scores, weights):
    order = np.argsort(scores)
    scores_sorted = scores[order].astype(np.float32)
    weights_sorted = weights[order].astype(np.float32)
    denom = float(np.sum(weights_sorted))
    if denom <= 0.0:
        return scores_sorted, np.zeros_like(scores_sorted, dtype=np.float32)
    cdf = (np.cumsum(weights_sorted) / denom).astype(np.float32)
    return scores_sorted, cdf


@torch.no_grad()
def fit_sgcp(backbone, assign_model, transform_loader, device="cpu", n_latent_samples=10):
    data = extract_all(backbone, transform_loader, device=device)
    scores = np.asarray(data["scores"], dtype=np.float32)
    weights = extract_sgcp_weights(
        assign_model, transform_loader, device=device, n_latent_samples=n_latent_samples
    ).astype(np.float32)
    weights = _normalize_rows(weights)

    global_scores, global_cdf = _weighted_ecdf(scores, np.ones_like(scores, dtype=np.float32))
    group_scores_sorted = []
    group_cdf_sorted = []
    for k in range(weights.shape[1]):
        if float(np.sum(weights[:, k])) <= 1e-8:
            group_scores_sorted.append(global_scores)
            group_cdf_sorted.append(global_cdf)
        else:
            s_sorted, cdf_sorted = _weighted_ecdf(scores, weights[:, k])
            group_scores_sorted.append(s_sorted)
            group_cdf_sorted.append(cdf_sorted)

    return SGCP(group_scores_sorted=group_scores_sorted, group_cdf_sorted=group_cdf_sorted, q_v=0.0)


@torch.no_grad()
def calibrate_sgcp_quantile(backbone, assign_model, sgcp_obj, cal_loader, alpha=0.1, device="cpu", n_latent_samples=10):
    data = extract_all(backbone, cal_loader, device=device)
    scores = np.asarray(data["scores"], dtype=np.float32)
    weights = extract_sgcp_weights(
        assign_model, cal_loader, device=device, n_latent_samples=n_latent_samples
    ).astype(np.float32)
    weights = _normalize_rows(weights)
    sgcp_obj.q_v = conformal_quantile(sgcp_obj.local_cdf(scores, weights), alpha)
    return sgcp_obj


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
    weights = _normalize_rows(weights)

    transformed = cp_obj.local_cdf(1.0 - probs, weights)
    pred_sets = [
        [label for label in range(probs.shape[1]) if float(transformed[i, label]) <= float(cp_obj.q_v)]
        for i in range(probs.shape[0])
    ]
    attrs, primary_attr, attr_meanings = get_eval_attrs(data, dataset_name)
    return evaluate_prediction_sets(
        pred_sets=pred_sets,
        y_true=data["y"],
        attrs=attrs,
        alpha=alpha,
        primary_attr=primary_attr,
        attr_meanings=attr_meanings,
    )
