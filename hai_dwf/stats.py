"""Corrected statistical analysis of Section 3.8."""
from __future__ import annotations
import numpy as np
from scipy import stats as ss

from .common import holm  # noqa: F401  (re-exported)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired per-instance McNemar test: exact binomial if b+c < 25, else chi-square with continuity correction."""
    a, b_ = correct_a.astype(bool), correct_b.astype(bool)
    b = int((a & ~b_).sum()); c = int((~a & b_).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "method": "none"}
    if n < 25:
        p = min(1.0, 2 * ss.binom.cdf(min(b, c), n, 0.5)); m = "exact"; chi = None
    else:
        chi = (abs(b - c) - 1) ** 2 / n; p = float(ss.chi2.sf(chi, 1)); m = "chi2_cc"
    return {"b": b, "c": c, "chi2": chi, "p": float(p), "method": m}


def cluster_bootstrap_diff(y, pred_a, pred_b, cluster, n_classes, B=10_000, seed=0, chunk=250) -> dict:
    """95% CI for the paired difference (A - B) in accuracy and macro-F1, resampling whole clusters (capture sessions / blocks)."""
    _, cl = np.unique(cluster, return_inverse=True); G = cl.max() + 1
    def per_cluster(pred):
        tp = np.zeros((G, n_classes)); fp = np.zeros((G, n_classes)); fn = np.zeros((G, n_classes))
        np.add.at(tp, (cl, y), (pred == y)); np.add.at(fp, (cl, pred), (pred != y)); np.add.at(fn, (cl, y), (pred != y))
        return tp, fp, fn
    A, Bc = per_cluster(pred_a), per_cluster(pred_b)
    n_g = np.bincount(cl, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    def f1_acc(w, T):
        tp, fp, fn = (w @ T[0]), (w @ T[1]), (w @ T[2])
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        present = (tp + fn) > 0
        return np.array([np.nanmean(np.where(p, f, np.nan)) for f, p in zip(f1, present)]), tp.sum(1) / (w @ n_g)
    d_f1, d_acc = [], []
    for a in range(0, B, chunk):
        m = min(chunk, B - a)
        w = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(m)]).astype(float)
        fa, aa = f1_acc(w, A); fb, ab = f1_acc(w, Bc)
        d_f1.append(100 * (fa - fb)); d_acc.append(100 * (aa - ab))
    d_f1, d_acc = np.concatenate(d_f1), np.concatenate(d_acc)
    return {"macro_f1_diff_ci95": np.percentile(d_f1, [2.5, 97.5]).tolist(), "acc_diff_ci95": np.percentile(d_acc, [2.5, 97.5]).tolist(),
            "n_clusters": int(G)}


# ----------------------------------------------------------------------------- DeLong (fast, paired, Sun & Xu 2014)
def _midrank(x):
    return ss.rankdata(x, method='average')


def delong_paired(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Two-sided DeLong test for two correlated binary ROC curves on the same instances."""
    pos = y_true.astype(bool); m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        return {"auc1": np.nan, "auc2": np.nan, "z": np.nan, "p": np.nan}
    S = np.stack([s1, s2]); k = 2
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        x, y = S[r][pos], S[r][~pos]
        tx[r], ty[r], tz[r] = _midrank(x), _midrank(y), _midrank(np.r_[x, y])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.cov(v01), np.cov(v10)
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": 0.0, "p": 1.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(2 * ss.norm.sf(abs(z)))}


def delong_ovr(y, P1, P2, classes=None) -> dict:
    """One-vs-rest DeLong per class + Holm adjustment across classes (the multi-class use justified in Section 3.8)."""
    C = P1.shape[1]; res = {}
    for c in (classes if classes is not None else range(C)):
        r = delong_paired((y == c), P1[:, c].astype(np.float64), P2[:, c].astype(np.float64))
        if not np.isnan(r["p"]):
            res[int(c)] = r
    keys = list(res); adj = holm([res[k]["p"] for k in keys]) if keys else []
    for k, a in zip(keys, adj):
        res[k]["p_holm"] = a
    return res


def wilcoxon_exact(diffs: np.ndarray) -> dict:
    """Paired two-sided exact Wilcoxon signed-rank over N seeds (min attainable p = 2 / 2^N)."""
    d = np.asarray(diffs, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return {"p": 1.0, "n": 0, "min_attainable_p": 2 / 2 ** len(d)}
    r = ss.wilcoxon(nz, alternative="two-sided", method="exact")
    return {"p": float(r.pvalue), "stat": float(r.statistic), "n": int(len(nz)), "min_attainable_p": 2 / 2 ** len(nz),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0}


def mean_sd_ci(x, level=0.95) -> dict:
    x = np.asarray(x, float); n = len(x)
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else 0.0
    h = ss.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(m), "sd": float(sd), "ci": [float(m - h), float(m + h)], "min": float(x.min()), "max": float(x.max()), "n": n}


def cohens_dz(diffs) -> float:
    d = np.asarray(diffs, float)
    return float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
