import math
import numpy as np
import torch.nn.functional as F
import torch
from SelectiveCI_fairness.cp_obj import *

def kl_diagonal_gaussians(mu_q, sig_q, mu_p, sig_p):
    var_q = sig_q ** 2
    var_p = sig_p ** 2
    kl = torch.log(sig_p / sig_q) + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5
    return kl.sum(dim=-1)

def loader_to_numpy(loader, device='cpu'):
    xs, ys, colors, ages, regions = [], [], [], [], []
    for batch in loader:
        # Handle both 5-element and 6-element batches
        if len(batch) == 5:
            x, y, color, age, region = batch
        elif len(batch) == 6:
            x, y, group1, group2, _attr1, attr2 = batch
            color = group1
            age = group2
            region = attr2
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
            
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
        colors.append(color.cpu().numpy())
        ages.append(age.cpu().numpy())
        regions.append(region.cpu().numpy())
    return {
        'x': np.concatenate(xs, axis=0),
        'y': np.concatenate(ys, axis=0),
        'color': np.concatenate(colors, axis=0),
        'age': np.concatenate(ages, axis=0),
        'region': np.concatenate(regions, axis=0),
    }


# =========================
# Conformal helpers
# =========================

@torch.no_grad()
def extract_all(backbone, loader, device="cpu"):
    backbone.eval()
    feats_all, probs_all, scores_all = [], [], []
    y_all, color_all, age_all, region_all = [], [], [], []

    for batch in loader:
        # Handle both 5-element and 6-element batches
        if len(batch) == 5:
            x, y, color, age, region = batch
        elif len(batch) == 6:
            x, y, group1, group2, _attr1, attr2 = batch
            color = group1
            age = group2
            region = attr2
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        
        x = x.to(device)
        logits, feats = backbone(x)
        probs = F.softmax(logits, dim=-1).cpu()
        scores = 1.0 - probs[torch.arange(len(y)), y]

        feats_all.append(feats.cpu().numpy())
        probs_all.append(probs.numpy())
        scores_all.append(scores.numpy())
        y_all.append(y.numpy())
        color_all.append(color.numpy())
        age_all.append(age.numpy())
        region_all.append(region.numpy())

    return {
        "feats": np.concatenate(feats_all),
        "probs": np.concatenate(probs_all),
        "scores": np.concatenate(scores_all),
        "y": np.concatenate(y_all),
        "color": np.concatenate(color_all),
        "age": np.concatenate(age_all),
        "region": np.concatenate(region_all),
    }


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def weighted_quantile(values, weights, q):
    values = np.asarray(values)
    weights = np.asarray(weights)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cum_weights = np.cumsum(weights)
    total = cum_weights[-1]
    if total <= 0:
        return float(np.max(values))
    idx = np.searchsorted(cum_weights, q * total, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])

def simple_kmeans(X, K=8, iters=50, seed=42):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=K, replace=False)].copy()

    for _ in range(iters):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        assign = np.argmin(dists, axis=1)

        new_centers = []
        for k in range(K):
            new_centers.append(centers[k] if np.sum(assign == k) == 0 else X[assign == k].mean(axis=0))
        centers = np.stack(new_centers)

    return centers


@torch.no_grad()
def extract_softproto_weights(model, loader, device="cpu"):
    model.eval()
    all_w = []
    for x, *_rest in loader:
        x = x.to(device)
        out = model(x)
        all_w.append(out["weights"].cpu().numpy())
    return np.concatenate(all_w)


@torch.no_grad()
def extract_sgcp_weights(model, loader, device="cpu", n_latent_samples=10):
    model.eval()
    all_w = []
    for x, *_rest in loader:
        x = x.to(device)
        out = model(x, n_latent_samples=n_latent_samples)
        all_w.append(out["avg_weights"].cpu().numpy())
    return np.concatenate(all_w)


@torch.no_grad()
def extract_sgcp_difficulties(model, loader, device="cpu", n_latent_samples=10):
    model.eval()
    all_d = []
    for x, *_rest in loader:
        x = x.to(device)
        out = model(x, n_latent_samples=n_latent_samples)
        all_d.append(out["difficulty"].cpu().numpy())
    return np.concatenate(all_d)



def prediction_set_from_probs_and_thresholds(probs, thresholds):
    pred_sets = []
    n, c = probs.shape
    for i in range(n):
        qx = thresholds[i]
        labels = [y for y in range(c) if (1.0 - probs[i, y]) <= qx]
        pred_sets.append(labels)
    return pred_sets
