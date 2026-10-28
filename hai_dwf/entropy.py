"""Window-level control signals for the DWF gate (Section 3.7) and the entropy controls of Section 3.9.

H(x) = -sum p ln p (nats) over the 28 x B histogram of a T-step window of unit-scaled HFR components, with p = count/(T*28).
Max = ln(28*B) (= ln 280 = 5.63 for B = 10).  IMPORTANT: because the 28 per-component histograms are pooled into one distribution, the MINIMUM is
ln 28 = 3.33 nats (every component constant), so the regime boundaries 0.6/1.4/2.2/3.0 printed in Table 7 cannot arise from this definition;
the pipeline therefore learns the boundaries on the validation partition inside the observed range (see fusion.tune_dwf).
"""
from __future__ import annotations
import numpy as np


def _hist(Wu: np.ndarray, bins: int) -> np.ndarray:
    n, T, F = Wu.shape
    b = np.minimum((Wu * bins).astype(np.int64), bins - 1)                  # (n,T,F)
    key = (np.arange(n)[:, None, None] * F + np.arange(F)[None, None, :]) * bins + b
    return np.bincount(key.ravel(), minlength=n * F * bins).reshape(n, F * bins).astype(np.float64)


def shannon(Wu: np.ndarray, bins: int = 10) -> np.ndarray:
    c = _hist(Wu, bins)
    p = c / c.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0.0).sum(1)


def renyi2(Wu, bins=10):
    c = _hist(Wu, bins); p = c / c.sum(1, keepdims=True)
    return -np.log((p ** 2).sum(1))


def tsallis2(Wu, bins=10):
    c = _hist(Wu, bins); p = c / c.sum(1, keepdims=True)
    return 1.0 - (p ** 2).sum(1)


def window_var(Wu: np.ndarray, comp: int) -> np.ndarray:
    return Wu[:, :, comp].var(axis=1)


SIGNAL_NAMES = ["shannon10", "shannon5", "shannon20", "shannon50", "renyi2", "tsallis2", "var_pkt"]


def all_signals(Wu: np.ndarray, pkt_comp: int, bins: int = 10) -> dict:
    return {
        "shannon10": shannon(Wu, bins), "shannon5": shannon(Wu, 5), "shannon20": shannon(Wu, 20),
        "shannon50": shannon(Wu, 50), "renyi2": renyi2(Wu, bins), "tsallis2": tsallis2(Wu, bins),
        "var_pkt": window_var(Wu, pkt_comp),
    }


def predictive_entropy(P_list: list[np.ndarray]) -> np.ndarray:
    """Instance-level uncertainty: entropy of the equal-weight mean posterior (variant B6)."""
    P = np.mean(P_list, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0.0).sum(1)


# ---- differentiable surrogate used by the gate-manipulation attack (Stage 3c) ------------------------
def soft_shannon_tf(Wu, bins: int = 10, bw: float = None):
    """Gaussian-kernel soft histogram entropy; Wu: (n,T,F) tensor in [0,1]."""
    import tensorflow as tf
    bw = bw or 0.5 / bins
    centers = (tf.range(bins, dtype=tf.float32) + 0.5) / bins
    d = (Wu[..., None] - centers) / bw                                  # (n,T,F,B)
    w = tf.nn.softmax(-0.5 * d ** 2, axis=-1)
    c = tf.reduce_sum(w, axis=1)                                        # (n,F,B)
    c = tf.reshape(c, (tf.shape(c)[0], -1))
    p = c / tf.reduce_sum(c, axis=1, keepdims=True)
    return -tf.reduce_sum(p * tf.math.log(p + 1e-12), axis=1)
