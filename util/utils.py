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
    xs, ys, attr1s, attr2s, attr3s, attr4s = [], [], [], [], [], []
    attr5s, attr6s, attr7s = [], [], []
    has_attr4 = False
    has_nursery_attrs = False
    for batch in loader:
        # Handle 5-element, 6-element, 7-element, and 9-element (nursery) batches
        if len(batch) == 5:
            # BACH / Single Sensitive / Synthetic
            x, y, a1, a2, a3 = batch
            a4 = torch.zeros_like(y)
        elif len(batch) == 6:
            # Check if this is synthetic data (which uses a1, a2 as groups, and a3 as something else)
            # or some other format. In two_sensitive, batch is (x, y, a1, a2, a_idx1, a_idx2)
            # Let's preserve the original behavior for two_sensitive
            x, y, a1, a2, _, a3 = batch
            a4 = torch.zeros_like(y)
        elif len(batch) == 7:
            x, y, a1, a2, a3, a4, _ = batch
            has_attr4 = True
        elif len(batch) == 9:
            x, y, a1, a2, a3, a4, a5, a6, a7 = batch
            has_attr4 = True
            has_nursery_attrs = True
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
            
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
        attr1s.append(a1.cpu().numpy())
        attr2s.append(a2.cpu().numpy())
        attr3s.append(a3.cpu().numpy())
        attr4s.append(a4.cpu().numpy())
        if has_nursery_attrs:
            attr5s.append(a5.cpu().numpy())
            attr6s.append(a6.cpu().numpy())
            attr7s.append(a7.cpu().numpy())
    
    out = {
        'x': np.concatenate(xs, axis=0),
        'y': np.concatenate(ys, axis=0),
        'color': np.concatenate(attr1s, axis=0),
        'age': np.concatenate(attr2s, axis=0),
        'region': np.concatenate(attr3s, axis=0),
    }
    if has_attr4:
        out['diag'] = np.concatenate(attr4s, axis=0)
    if has_nursery_attrs:
        out['attr5'] = np.concatenate(attr5s, axis=0)
        out['attr6'] = np.concatenate(attr6s, axis=0)
        out['attr7'] = np.concatenate(attr7s, axis=0)
    return out


# =========================
# Conformal helpers
# =========================
import torch
import numpy as np

@torch.no_grad()
def extract_all(backbone, loader, device="cpu"):
    backbone.eval()
    backbone.to(device)

    all_y, all_probs, all_scores, all_feats = [], [], [], []
    all_attrs = {f"attr{i+1}": [] for i in range(7)}

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        logits, feats = backbone(x)
        probs = torch.softmax(logits, dim=1)
        scores = 1.0 - probs.gather(1, y.unsqueeze(1)).squeeze(1)

        all_y.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
        all_feats.append(feats.cpu().numpy())

        # MIMIC/Syn/BACH 5  Nursery 9
        for i in range(2, len(batch)):
            attr_key = f"attr{i-1}"
            if attr_key in all_attrs:
                all_attrs[attr_key].append(batch[i].cpu().numpy())

    data = {
        "y": np.concatenate(all_y),
        "probs": np.concatenate(all_probs),
        "scores": np.concatenate(all_scores),
        "feats": np.concatenate(all_feats),
    }

    for attr_key, attr_list in all_attrs.items():
        if len(attr_list) > 0:
            data[attr_key] = np.concatenate(attr_list)

    return data

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
def extract_proto_weights(model, loader, device="cpu"):
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
