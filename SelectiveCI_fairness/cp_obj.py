
from dataclasses import dataclass
import numpy as np

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
class HardClusterCP:
    cluster_thresholds: np.ndarray
    cluster_centers: np.ndarray

    def threshold_for_batch(self, feats):
        dists = ((feats[:, None, :] - self.cluster_centers[None, :, :]) ** 2).sum(axis=-1)
        c = np.argmin(dists, axis=1)
        return self.cluster_thresholds[c]


@dataclass
class SoftPrototypeCP:
    prototype_thresholds: np.ndarray
    mode: str = "top1"
    gamma: float = 2.0

    def threshold_for_batch(self, weights):
        if self.mode == "avg":
            return np.sum(weights * self.prototype_thresholds[None, :], axis=1)
        if self.mode == "sharpened_avg":
            sharp_w = np.power(np.clip(weights, 1e-12, 1.0), self.gamma)
            sharp_w = sharp_w / sharp_w.sum(axis=1, keepdims=True)
            return np.sum(sharp_w * self.prototype_thresholds[None, :], axis=1)
        top_idx = np.argmax(weights, axis=1)
        return self.prototype_thresholds[top_idx]


@dataclass
class BinnedSLSCP:
    bin_edges: np.ndarray
    bin_thresholds: np.ndarray
    fallback_threshold: float

    def threshold_for_batch(self, difficulties):
        difficulties = np.asarray(difficulties, dtype=np.float32)
        idx = np.digitize(difficulties, self.bin_edges[1:-1], right=False)
        idx = np.clip(idx, 0, len(self.bin_thresholds) - 1)
        return self.bin_thresholds[idx]

