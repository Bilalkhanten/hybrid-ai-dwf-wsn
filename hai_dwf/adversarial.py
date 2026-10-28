"""Adversarial feasibility protocol (Section 3.11, Stages 1 and 3; Stage 2 = replay.py).

Perturbations act on the scaled 28-component HFR windows (T x 28).  "Unconstrained" attacks may change every component;
"constrained" attacks change only attacker-controllable components (regex in the config), respect physical bounds
(non-negativity, integer counts) and re-derive engineered components from their perturbed sources (dependency projection)
after every step.  White-box attacks use CNN + BiLSTM as a differentiable substitute and are transferred to the full
ensemble (GBDT + DWF gate).  Stage 3 adds a decision-only attack on the full ensemble, a tree-evasion attack on the
LightGBM component and an EOT gate-manipulation attack through a smooth surrogate of the regime lookup.

    python -m hai_dwf.adversarial --cfg config/default.yaml --dataset ciciot2023 --seed 0
"""
from __future__ import annotations
import argparse, pickle, re
from pathlib import Path
import numpy as np
import tensorflow as tf

from .common import load_cfg, run_dir, stage_seed, write_json, read_json, seed_all
from . import data as D, entropy as H, fusion as Fu
from .models import TemporalAttention  # noqa: F401  (registers the custom layer)


# ----------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg, ds, seed):
        import lightgbm as lgb
        self.cfg, self.ds, self.seed = cfg, ds, seed
        rd = self.rd = run_dir(cfg, ds, seed)
        with open(rd / "preproc.pkl", "rb") as f:
            o = pickle.load(f)
        self.pp, self.spec, self.pkt = o["pp"], o["spec"], o["pkt_comp"]
        self.cnn = tf.keras.models.load_model(rd / "cnn.keras"); self.bil = tf.keras.models.load_model(rd / "bilstm.keras")
        self.gb = lgb.Booster(model_file=str(rd / "lgbm.txt"))
        self.dwf = read_json(rd / "dwf_params.json")["dwf"]
        df = D.load_raw(cfg, ds); dc = cfg["datasets"][ds]
        P = np.load(rd / "partitions.npz"); rows = np.flatnonzero(P["part"] == 2)
        Zt = self.pp.transform(D.apply_hfr(df.iloc[rows], self.spec))
        idx = D.window_index(P["block"][rows], cfg["window"]); y = P["y"][rows]
        rng = np.random.default_rng(stage_seed(seed, "adversarial"))
        n = min(cfg["adversarial"]["n_eval"], len(rows)); ix = np.sort(rng.permutation(len(rows))[:n])
        self.W0 = Zt[idx[ix]].astype(np.float32); self.y = y[ix]
        self.C = self.cnn.output_shape[-1]
        self.names = [s["name"] for s in self.spec]
        a = cfg["adversarial"]
        ctrl = np.array([bool(re.search(a["controllable_regex"], n, re.I)) and not re.search(a["non_controllable_regex"], n, re.I)
                         and s["provenance"] == "raw" for n, s in zip(self.names, self.spec)])
        # engineered components are dependents: re-derived if all their sources are controllable, otherwise frozen
        self.dep = {}
        for j, s in enumerate(self.spec):
            if s["provenance"] == "engineered":
                src = [self.names.index(x) for x in s["source"] if x in self.names]
                if src and all(ctrl[i] for i in src):
                    self.dep[j] = src
        self.mask = ctrl.astype(np.float32)
        self.int_cols = np.array([bool(re.search(r"(count|conn|pkts|packets|hop|num)", n, re.I)) for n in self.names])
        self.nonneg = self.pp.raw_min >= 0

    # --- ensemble
    def posteriors(self, W):
        W = np.asarray(W, np.float32)
        pc = self.cnn.predict(W[:, -1, :], batch_size=8192, verbose=0)
        pb = self.bil.predict(W, batch_size=4096, verbose=0)
        pg = self.gb.predict(W[:, -1, :])
        return np.stack([pc, pb, pg], 1).astype(np.float32)

    def predict(self, W):
        Pst = self.posteriors(W)
        sig = H.shannon(self.pp.unit(W), self.cfg["bins"])
        return Fu.apply_dwf(self.dwf, Pst, sig)

    def acc(self, W, comp=None):
        if comp == "gbdt":
            return float((self.gb.predict(W[:, -1, :]).argmax(1) == self.y).mean() * 100)
        return float((self.predict(W).argmax(1) == self.y).mean() * 100)

    # --- constraint projection (Stage 1)
    def project(self, Wp, W0, eps, norm="linf", constrained=True):
        d = Wp - W0
        if constrained:
            d = d * self.mask[None, None, :]
        if norm == "linf":
            d = np.clip(d, -eps, eps)
        else:
            nrm = np.sqrt((d ** 2).sum((1, 2), keepdims=True)) + 1e-12
            d = d * np.minimum(1.0, eps / nrm)
        W = np.clip(W0 + d, -1.0, 1.0)
        if constrained:
            raw = W * self.pp.iqr + self.pp.med
            raw = np.where(self.nonneg[None, None, :], np.maximum(raw, 0.0), raw)
            raw = np.where(self.int_cols[None, None, :], np.round(raw), raw)
            for j, src in self.dep.items():                     # dependency projection of engineered components
                s = self.spec[j]
                if s["op"] == "log1p_signed":
                    raw[..., j] = np.sign(raw[..., src[0]]) * np.log1p(np.abs(raw[..., src[0]]))
                elif s["op"] == "ratio":
                    den = raw[..., src[1]]; raw[..., j] = raw[..., src[0]] / np.where(np.abs(den) < 1e-9, np.nan, den)
            raw = np.nan_to_num(raw)
            W = np.clip((raw - self.pp.med) / self.pp.iqr, -1.0, 1.0)
        return W.astype(np.float32)


def _surrogate_logp(ctx, W):
    pc = ctx.cnn(W[:, -1, :], training=False); pb = ctx.bil(W, training=False)
    return tf.math.log(0.5 * (pc + pb) + 1e-9)


def _grad(ctx, W, y, fn):
    W = tf.convert_to_tensor(W)
    with tf.GradientTape() as t:
        t.watch(W); loss = fn(W, y)
    return t.gradient(loss, W).numpy()


def _ce(ctx):
    return lambda W, y: tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, tf.exp(_surrogate_logp(ctx, W))))


def _batches(n, b=2048):
    for a in range(0, n, b):
        yield slice(a, min(a + b, n))


# ----------------------------------------------------------------------------- attacks
def fgsm(ctx, eps, constrained):
    out = np.empty_like(ctx.W0)
    for sl in _batches(len(ctx.y)):
        g = _grad(ctx, ctx.W0[sl], ctx.y[sl], _ce(ctx))
        out[sl] = ctx.project(ctx.W0[sl] + eps * np.sign(g), ctx.W0[sl], eps, "linf", constrained)
    return out


def pgd(ctx, eps, constrained, steps):
    out = np.empty_like(ctx.W0); rng = np.random.default_rng(stage_seed(ctx.seed, "adversarial"))
    for sl in _batches(len(ctx.y)):
        W0 = ctx.W0[sl]; W = ctx.project(W0 + rng.uniform(-eps, eps, W0.shape).astype(np.float32), W0, eps, "linf", constrained)
        for _ in range(steps):
            g = _grad(ctx, W, ctx.y[sl], _ce(ctx))
            W = ctx.project(W + 2.5 * eps / steps * np.sign(g), W0, eps, "linf", constrained)
        out[sl] = W
    return out


def _margin(ctx):
    def f(W, y):
        lp = _surrogate_logp(ctx, W); oh = tf.one_hot(y, ctx.C)
        true = tf.reduce_sum(lp * oh, 1); other = tf.reduce_max(lp - 1e9 * oh, 1)
        return tf.reduce_sum(true - other)
    return f


def deepfool(ctx, eps, constrained, steps=20, overshoot=0.02):
    """Minimum-L2 attack on the surrogate margin; the perturbation is finally clipped to the L2 ball of radius eps."""
    out = np.empty_like(ctx.W0)
    for sl in _batches(len(ctx.y), 1024):
        W0 = ctx.W0[sl]; W = W0.copy(); y = ctx.y[sl]; r = np.zeros_like(W0)
        for _ in range(steps):
            lp = _surrogate_logp(ctx, tf.convert_to_tensor(W)).numpy()
            oh = np.eye(ctx.C)[y]; marg = (lp * oh).sum(1) - (lp - 1e9 * oh).max(1)
            act = marg > 0
            if not act.any():
                break
            g = _grad(ctx, W, y, _margin(ctx)) * act[:, None, None]
            gn = (g ** 2).sum((1, 2), keepdims=True) + 1e-12
            r = r - (marg[:, None, None] * g / gn) * (1 + overshoot)
            W = ctx.project(W0 + r, W0, eps, "l2", constrained)
        out[sl] = W
    return out


def cw_l2(ctx, eps, constrained, steps=20, c=1.0, lr=0.05, kappa=0.0):
    out = np.empty_like(ctx.W0)
    for sl in _batches(len(ctx.y), 1024):
        W0 = ctx.W0[sl]; y = ctx.y[sl]; d = np.zeros_like(W0)
        m = np.zeros_like(d); v = np.zeros_like(d)
        for t in range(1, steps + 1):
            W = ctx.project(W0 + d, W0, eps, "l2", constrained)
            Wt = tf.convert_to_tensor(W)
            with tf.GradientTape() as tp:
                tp.watch(Wt); lp = _surrogate_logp(ctx, Wt); oh = tf.one_hot(y, ctx.C)
                gap = tf.reduce_sum(lp * oh, 1) - tf.reduce_max(lp - 1e9 * oh, 1)
                loss = tf.reduce_sum(tf.square(Wt - W0)) + c * tf.reduce_sum(tf.maximum(gap, -kappa))
            g = tp.gradient(loss, Wt).numpy()
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
            d = d - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        out[sl] = ctx.project(W0 + d, W0, eps, "l2", constrained)
    return out


def square_blackbox(ctx, eps, constrained, queries):
    """Decision-only random-search (Square-type) L_inf attack on the COMPLETE ensemble (GBDT + DWF gate included)."""
    rng = np.random.default_rng(stage_seed(ctx.seed, "adversarial"))
    W0 = ctx.W0; n, T, F = W0.shape
    feat = np.flatnonzero(ctx.mask > 0) if constrained else np.arange(F)
    if len(feat) == 0:
        return W0.copy()
    W = ctx.project(W0 + eps * rng.choice([-1, 1], (n, 1, F)).astype(np.float32), W0, eps, "linf", constrained)
    best = ctx.predict(W)[np.arange(n), ctx.y]
    for _ in range(max(1, queries // 10)):                 # each iteration = one query per example; 10 candidate patches batched
        W_new = W.copy()
        t0 = rng.integers(0, T, n); f0 = feat[rng.integers(0, len(feat), n)]
        sign = rng.choice([-1, 1], n)
        ar = np.arange(n); tmask = np.arange(T)[None, :] >= t0[:, None]
        cand = np.clip(W0[ar[:, None], np.arange(T)[None, :], f0[:, None]] + sign[:, None] * eps, -1, 1)
        cur = W_new[ar[:, None], np.arange(T)[None, :], f0[:, None]]
        W_new[ar[:, None], np.arange(T)[None, :], f0[:, None]] = np.where(tmask, cand, cur)
        W_new = ctx.project(W_new, W0, eps, "linf", constrained)
        p = ctx.predict(W_new)[np.arange(n), ctx.y]
        better = p < best
        W[better] = W_new[better]; best = np.where(better, p, best)
    return W


def tree_evasion(ctx, eps, constrained, steps=10):
    """Greedy threshold-crossing evasion of the LightGBM component on the last time step (L_inf <= eps in scaled space)."""
    df = ctx.gb.trees_to_dataframe(); df = df[df.split_feature.notna()]
    feats = df.split_feature.str.replace("Column_", "", regex=False)
    thr = {}
    for f, g in df.groupby(feats):
        thr[int(f)] = np.unique(g.threshold.to_numpy())
    W = ctx.W0.copy(); n = len(W); x0 = ctx.W0[:, -1, :]
    allowed = [f for f in range(x0.shape[1]) if (ctx.mask[f] > 0 or not constrained) and f in thr]
    active = np.ones(n, bool)
    for _ in range(steps):
        pt = ctx.gb.predict(W[:, -1, :])[np.arange(n), ctx.y]
        active &= ctx.gb.predict(W[:, -1, :]).argmax(1) == ctx.y
        if not active.any():
            break
        bestp, bestW = pt.copy(), W[:, -1, :].copy()
        for f in allowed:
            th = thr[f]
            for direction in (-1, 1):
                cur = W[:, -1, f]
                pos = np.searchsorted(th, cur)
                j = np.clip(pos - 1 if direction < 0 else pos, 0, len(th) - 1)
                cand = th[j] - 1e-6 if direction < 0 else th[j] + 1e-6
                cand = np.clip(cand, x0[:, f] - eps, x0[:, f] + eps)
                trial = W[:, -1, :].copy(); trial[:, f] = np.clip(cand, -1, 1)
                p = ctx.gb.predict(trial)[np.arange(n), ctx.y]
                better = active & (p < bestp - 1e-9)
                bestp[better] = p[better]; bestW[better] = trial[better]
        W[:, -1, :] = np.where(active[:, None], bestW, W[:, -1, :])
    W = ctx.project(W, ctx.W0, eps, "linf", constrained)
    return W


def gate_manipulation(ctx, eps, constrained, steps, eot=4, sigma=0.01):
    """PGD through a smooth surrogate of the entropy-indexed gate (expectation over transformation).  GBDT posteriors are held at
    their value on the current iterate (non-differentiable); CNN/BiLSTM and the gate are differentiated."""
    par = ctx.dwf; b = np.sort(np.array(par["bounds"], np.float32)); Wm = tf.constant(np.array(par["W"], np.float32))
    tau = max(0.05, float(np.std(b)) * 0.1 + 1e-3) if len(b) else 1.0
    pp = ctx.pp; mn, mx = tf.constant(pp.mn), tf.constant(pp.mx)
    out = np.empty_like(ctx.W0); rng = np.random.default_rng(stage_seed(ctx.seed, "adversarial"))

    def soft_fused(W, pg):
        U = tf.clip_by_value((W - mn) / (mx - mn), 0.0, 1.0)
        s = H.soft_shannon_tf(U, ctx.cfg["bins"])
        edges = tf.concat([[-1e9], b, [1e9]], 0)
        r = tf.sigmoid((s[:, None] - edges[None, :-1]) / tau) - tf.sigmoid((s[:, None] - edges[None, 1:]) / tau)  # (n,K)
        w = tf.matmul(r, Wm)                                                                                       # (n,3)
        p = tf.stack([ctx.cnn(W[:, -1, :], training=False), ctx.bil(W, training=False), pg], 1)
        return tf.reduce_sum(w[:, :, None] * p, 1)

    for sl in _batches(len(ctx.y), 1024):
        W0 = ctx.W0[sl]; y = ctx.y[sl]; W = W0.copy()
        for _ in range(steps):
            pg = tf.constant(ctx.gb.predict(W[:, -1, :]).astype(np.float32)); g = np.zeros_like(W)
            for _e in range(eot):
                Wn = tf.convert_to_tensor(W + rng.normal(0, sigma, W.shape).astype(np.float32))
                with tf.GradientTape() as t:
                    t.watch(Wn); pf = soft_fused(Wn, pg)
                    loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, pf))
                g += t.gradient(loss, Wn).numpy() / eot
            W = ctx.project(W + 2.5 * eps / steps * np.sign(g), W0, eps, "linf", constrained)
        out[sl] = W
    return out


# ----------------------------------------------------------------------------- driver
def run(cfg, ds, seed, force=False):
    rd = run_dir(cfg, ds, seed)
    if (rd / "adversarial.json").exists() and not force:
        print("[skip]", rd / "adversarial.json"); return
    seed_all(stage_seed(seed, "adversarial"))
    ctx = Ctx(cfg, ds, seed); a = cfg["adversarial"]
    res = {"clean_acc": ctx.acc(ctx.W0), "clean_acc_gbdt": ctx.acc(ctx.W0, "gbdt"), "n_eval": int(len(ctx.y)),
           "controllable_components": [n for n, m in zip(ctx.names, ctx.mask) if m > 0], "attacks": {}}
    for eps in a["eps_grid"]:
        for cons in (False, True):
            tag = "constrained" if cons else "unconstrained"
            for name, fn in (("FGSM", lambda e, c: fgsm(ctx, e, c)), ("PGD", lambda e, c: pgd(ctx, e, c, a["pgd_steps"])),
                             ("DeepFool", lambda e, c: deepfool(ctx, e, c)), ("CW", lambda e, c: cw_l2(ctx, e, c)),
                             ("BlackBox", lambda e, c: square_blackbox(ctx, e, c, a["query_budget"])),
                             ("TreeEvasion", lambda e, c: tree_evasion(ctx, e, c)),
                             ("GateManip", lambda e, c: gate_manipulation(ctx, e, c, a["pgd_steps"]))):
                Wa = fn(eps, cons)
                r = res["attacks"].setdefault(name, {}).setdefault(str(eps), {})
                r[tag] = {"ensemble_acc": ctx.acc(Wa), "gbdt_acc": ctx.acc(Wa, "gbdt"),
                          "mean_linf": float(np.abs(Wa - ctx.W0).max((1, 2)).mean()),
                          "mean_l2": float(np.sqrt(((Wa - ctx.W0) ** 2).sum((1, 2))).mean())}
                print(f"{ds} s{seed} {name} eps={eps} {tag}: {r[tag]['ensemble_acc']:.2f}")
    write_json(rd / "adversarial.json", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); run(load_cfg(a.cfg), a.dataset, a.seed, a.force)


if __name__ == "__main__":
    main()
