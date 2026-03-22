
from dataclasses import dataclass
import numpy as np
from typing import List

# =========================
# CP objects
# =========================
@dataclass
class GlobalCP:
    threshold: float

@dataclass
class FixedGroupCP:
    thresholds: dict
    fallback_threshold: float

    def threshold_for_keys(self, keys):
        return np.array([self.thresholds.get(k, self.fallback_threshold) for k in keys], dtype=np.float32)


@dataclass
class ClusteredCP:
    cluster_thresholds: np.ndarray
    cluster_centers: np.ndarray

    def threshold_for_batch(self, feats):
        dists = ((feats[:, None, :] - self.cluster_centers[None, :, :]) ** 2).sum(axis=-1)
        c = np.argmin(dists, axis=1)
        return self.cluster_thresholds[c]


@dataclass
class PrototypeCP:
    prototype_thresholds: np.ndarray
    mode: str = "top1"
    gamma: float = 2.0

    def threshold_for_batch(self, weights):
        if self.mode == "avg":
            return np.sum(weights * self.prototype_thresholds[None, :], axis=1)
        top_idx = np.argmax(weights, axis=1)
        return self.prototype_thresholds[top_idx]



def _eval_step_ecdf(scores_sorted: np.ndarray, cdf_sorted: np.ndarray, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float32)
    idx = np.searchsorted(scores_sorted, s, side="right") - 1
    out = np.zeros_like(s, dtype=np.float32)
    if len(cdf_sorted) == 0:
        return out
    idx_clip = np.clip(idx, 0, len(cdf_sorted) - 1)
    out = cdf_sorted[idx_clip]
    out[idx < 0] = 0.0
    return out


@dataclass
class SGCP:
    group_scores_sorted: List[np.ndarray]
    group_cdf_sorted: List[np.ndarray]
    q_v: float

    def local_cdf(self, s: np.ndarray, weights: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim != 2:
            raise ValueError(f"weights must be (n,k), got {weights.shape}")
        if s.shape[0] != weights.shape[0]:
            raise ValueError(f"s first dim must match weights n, got s={s.shape}, weights={weights.shape}")

        k = weights.shape[1]
        if k != len(self.group_scores_sorted):
            raise ValueError(f"weights k must match groups, got k={k}, groups={len(self.group_scores_sorted)}")

        out = np.zeros_like(s, dtype=np.float32)
        for g in range(k):
            fg = _eval_step_ecdf(self.group_scores_sorted[g], self.group_cdf_sorted[g], s)
            if s.ndim == 1:
                out += weights[:, g] * fg
            else:
                out += weights[:, g, None] * fg
        return out
