"""Unit tests of the statistics / entropy / windowing code (run: python tests/test_units.py)."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy import stats as ss
from sklearn.metrics import roc_auc_score
from hai_dwf import stats as S, entropy as H, data as D
from hai_dwf.common import holm, evaluate

rng = np.random.default_rng(0)
# DeLong AUCs equal sklearn's; identical scores -> p = 1
y = rng.integers(0, 2, 2000); s1 = y * 0.8 + rng.normal(0, 1, 2000); s2 = y * 0.6 + rng.normal(0, 1, 2000)
r = S.delong_paired(y, s1, s2)
assert abs(r["auc1"] - roc_auc_score(y, s1)) < 1e-9 and abs(r["auc2"] - roc_auc_score(y, s2)) < 1e-9, r
assert S.delong_paired(y, s1, s1)["p"] == 1.0
# z = 2.9 two-sided p ~ 0.0037 (the correction made in Section 3.8)
assert abs(2 * ss.norm.sf(2.9) - 0.00373) < 1e-4
# exact Wilcoxon: N = 5 min p = 0.0625, N = 10 min p = 0.001953
assert abs(S.wilcoxon_exact(np.arange(1, 6))["p"] - 0.0625) < 1e-9
assert abs(S.wilcoxon_exact(np.arange(1, 11))["p"] - 2 / 1024) < 1e-9
# Holm
assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
# McNemar exact vs chi2
a = np.array([1] * 30 + [0] * 5 + [1] * 5, bool); b = np.array([1] * 30 + [1] * 5 + [0] * 5, bool)
m = S.mcnemar(a, b); assert m["b"] == 5 and m["c"] == 5 and m["p"] > 0.9
# Shannon entropy: constant window -> 0 ; maximal spread -> <= ln(280)
Wc = np.zeros((1, 20, 28), np.float32); assert abs(H.shannon(Wc)[0] - math.log(28)) < 1e-9   # NOTE: pooled definition => min H = ln 28 = 3.33 nats, not 0
Wr = rng.random((200, 20, 28)).astype(np.float32); h = H.shannon(Wr)
assert h.max() <= math.log(280) + 1e-9 and h.mean() > 4.0, h.mean()
assert abs(H.tsallis2(Wc)[0] - (1 - 1 / 28)) < 1e-9
# windows never cross block boundaries
blk = np.array([0] * 5 + [1] * 7); idx = D.window_index(blk, 4)
assert (blk[idx] == blk[:, None]).all()
assert (idx[0] == 0).all() and idx[4, -1] == 4 and idx[5, 0] == 5
# metrics
cm = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
assert abs(cm["accuracy"] - 75) < 1e-9
# cluster bootstrap CI contains the point estimate for a clearly better model
yy = rng.integers(0, 3, 3000); cl = np.repeat(np.arange(60), 50)
pa = np.where(rng.random(3000) < 0.95, yy, (yy + 1) % 3); pb = np.where(rng.random(3000) < 0.85, yy, (yy + 1) % 3)
ci = S.cluster_bootstrap_diff(yy, pa, pb, cl, 3, B=300)
assert ci["acc_diff_ci95"][0] > 0, ci
print("all unit tests passed")
