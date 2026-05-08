import math

import numpy as np
import torch


@torch.no_grad()
def extract_all(backbone, loader, device="cpu"):
    backbone.eval()
    backbone.to(device)

    ys, probs, scores, feats = [], [], [], []
    attrs = {f"attr{i}": [] for i in range(1, 8)}

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits, h = backbone(x)
        p = torch.softmax(logits, dim=1)
        s = 1.0 - p.gather(1, y.unsqueeze(1)).squeeze(1)

        ys.append(y.cpu().numpy())
        probs.append(p.cpu().numpy())
        scores.append(s.cpu().numpy())
        feats.append(h.cpu().numpy())

        for i in range(2, len(batch)):
            key = f"attr{i - 1}"
            if key in attrs:
                attrs[key].append(batch[i].cpu().numpy())

    data = {
        "y": np.concatenate(ys),
        "probs": np.concatenate(probs),
        "scores": np.concatenate(scores),
        "feats": np.concatenate(feats),
    }
    for key, values in attrs.items():
        if values:
            data[key] = np.concatenate(values)
    return data


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


@torch.no_grad()
def extract_sgcp_weights(model, loader, device="cpu", n_latent_samples=10):
    model.eval()
    weights = []
    for x, *_rest in loader:
        out = model(x.to(device), n_latent_samples=n_latent_samples)
        weights.append(out["avg_weights"].cpu().numpy())
    return np.concatenate(weights)
