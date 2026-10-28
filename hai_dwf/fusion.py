"""Stage 2: every fusion variant of Section 3.9 on the stored component posteriors (identical inputs for all variants).

    python -m hai_dwf.fusion --cfg config/default.yaml --dataset ciciot2023 --seed 0
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np

from .common import load_cfg, seed_all, stage_seed, run_dir, write_json, evaluate
from . import entropy as H

COMP3 = ["cnn", "bilstm", "gbdt"]


# ----------------------------------------------------------------------------- load
def load_comp(rd: Path) -> dict:
    z = np.load(rd / "components.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def stack3(Z: dict, split: str) -> np.ndarray:
    return np.stack([Z[f"p_{c}_{split}"] for c in COMP3], 1).astype(np.float32)          # (n,3,C)


# ----------------------------------------------------------------------------- DWF core
def regime_of(sig: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(bounds), sig, side="right")


def fuse(Pst: np.ndarray, sig: np.ndarray, bounds: np.ndarray, W: np.ndarray) -> np.ndarray:
    r = regime_of(sig, bounds) if len(bounds) else np.zeros(len(sig), int)
    return np.einsum("nk,nkc->nc", W[r], Pst)


def _norm(w):
    w = np.asarray(w, float) + 1e-6
    return w / w.sum(-1, keepdims=True)


def subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).permutation(n)[:cap])


def tune_dwf(sets: list[dict], K: int, seed: int, trials: int, free_bounds=True, fixed_bounds=None, cap=200_000,
             tag="dwf") -> dict:
    """sets: [{'P': (n,3,C), 'y': (n,), 'sig': (n,), 'C': int}] -- objective = mean validation macro-F1 over the sets
    (one set for P1/P2, three sets for the pooled leave-one-dataset-out search P3).  K = number of regimes (1 = fixed weights)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sub = []
    for i, s in enumerate(sets):
        ix = subsample(len(s["y"]), cap, stage_seed(seed, "tpe") + i)
        sub.append((s["P"][ix], s["y"][ix], s["sig"][ix], s["C"]))
    lo = min(np.percentile(s[2], 1) for s in sub); hi = max(np.percentile(s[2], 99) for s in sub)
    if hi <= lo:
        hi = lo + 1e-6

    def build(tr):
        if K == 1:
            b = np.array([])
        elif free_bounds:
            b = np.sort([tr.suggest_float(f"b{i}", lo, hi) for i in range(K - 1)])
        else:
            b = np.asarray(fixed_bounds, float)
        W = _norm([[tr.suggest_float(f"w{k}_{j}", 0.0, 1.0) for j in range(3)] for k in range(K)])
        return b, W

    def score(b, W):
        return float(np.mean([evaluate(y, fuse(P, s, b, W).argmax(1), C)["macro_f1"] for P, y, s, C in sub]))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=stage_seed(seed, "tpe")))
    st.optimize(lambda tr: score(*build(tr)), n_trials=trials)
    b, W = build(optuna.trial.FixedTrial(st.best_params))
    return {"bounds": np.asarray(b).tolist(), "W": W.tolist(), "val_macro_f1": st.best_value, "trials": trials, "K": K, "tag": tag}


def apply_dwf(par: dict, Pst, sig):
    return fuse(Pst, sig, np.array(par["bounds"]), np.array(par["W"]))


# ----------------------------------------------------------------------------- learned fusion baselines
def stacking(Z, cfg, seed, trials):
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    C = Z["p_cnn_val"].shape[1]
    ix = subsample(len(Z["y_val"]), min(cfg["tune_rows"]["fusion"], 50_000), stage_seed(seed, "shuffle"))
    Xv = np.log(np.clip(np.concatenate([Z[f"p_{c}_val"] for c in COMP3], 1), 1e-6, 1))[ix]; yv = Z["y_val"][ix]
    Xt = np.log(np.clip(np.concatenate([Z[f"p_{c}_test"] for c in COMP3], 1), 1e-6, 1))

    def obj(tr):
        m = LogisticRegression(C=tr.suggest_float("C", 1e-3, 1e3, log=True), max_iter=200)
        n = len(yv); h = n // 2
        m.fit(Xv[:h], yv[:h])
        pr = m.predict(Xv[h:])
        return evaluate(yv[h:], pr, C)["macro_f1"]
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=stage_seed(seed, "tpe")))
    st.optimize(obj, n_trials=trials)
    m = LogisticRegression(C=st.best_params["C"], max_iter=300).fit(Xv, yv)
    P = np.zeros((len(Xt), C), np.float32); P[:, m.classes_] = m.predict_proba(Xt)
    return P, {"val_macro_f1": st.best_value, "trials": trials, **st.best_params}


def learned_gate(Z, cfg, seed, trials, use_hfr: bool):
    import optuna, tensorflow as tf
    from . import models as M
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    C = Z["p_cnn_val"].shape[1]
    ix = subsample(len(Z["y_val"]), cfg["tune_rows"]["fusion"], stage_seed(seed, "shuffle"))
    s = Z["sig_shannon10_val"]; mu, sd = s.mean(), s.std() + 1e-9

    def feats(split, ix=None):
        sg = ((Z[f"sig_shannon10_{split}"] - mu) / sd)[:, None]
        X = np.concatenate([sg, Z[f"hfr_{split}"]], 1) if use_hfr else sg
        X = X.astype(np.float32)
        return X if ix is None else X[ix]
    Xg = feats("val", ix); P3 = stack3(Z, "val")[ix]; yv = Z["y_val"][ix]
    n = len(yv); h = int(0.8 * n)
    best = {"f1": -1, "model": None}

    def make(hidden, wd):
        g = tf.keras.layers.Input((Xg.shape[1],)); p = tf.keras.layers.Input((3, C))
        w = M.build_gate_mlp(Xg.shape[1], hidden, wd)(g)
        out = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z[0][:, :, None] * z[1], axis=1))([w, p])
        return tf.keras.Model([g, p], out)

    def obj(tr):
        tf.keras.utils.set_random_seed(stage_seed(seed, "init"))
        m = make(tr.suggest_categorical("hidden", [8, 16, 32, 64]), tr.suggest_float("wd", 1e-8, 1e-2, log=True))
        m.compile(tf.keras.optimizers.Adam(tr.suggest_float("lr", 1e-4, 1e-2, log=True)),
                  loss=lambda yt, yp: tf.keras.losses.sparse_categorical_crossentropy(yt, tf.clip_by_value(yp, 1e-7, 1.0)))
        m.fit([Xg[:h], P3[:h]], yv[:h], epochs=cfg["nn"]["epochs"] if cfg["nn"]["epochs"] < 30 else 30, batch_size=cfg["nn"]["batch"], verbose=0,
              validation_data=([Xg[h:], P3[h:]], yv[h:]),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        f1 = evaluate(yv[h:], m.predict([Xg[h:], P3[h:]], verbose=0).argmax(1), C)["macro_f1"]
        if f1 > best["f1"]:
            best.update(f1=f1, model=m)
        return f1
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=stage_seed(seed, "tpe")))
    st.optimize(obj, n_trials=trials)
    Pt = best["model"].predict([feats("test"), stack3(Z, "test")], batch_size=8192, verbose=0)
    return Pt.astype(np.float32), {"val_macro_f1": st.best_value, "trials": trials, **st.best_params}


def dempster_shafer(Z, split, alphas):
    P = [Z[f"p_{c}_{split}"].astype(np.float64) for c in COMP3]
    C = P[0].shape[1]
    a = alphas[0] * P[0]; th = np.full(len(a), 1 - alphas[0])
    for i in (1, 2):
        b = alphas[i] * P[i]; thb = np.full(len(b), 1 - alphas[i])
        k = a.sum(1) * b.sum(1) - (a * b).sum(1)
        den = np.clip(1 - k, 1e-9, None)[:, None]
        a, th = (a * b + a * thb[:, None] + th[:, None] * b) / den, th * thb / den[:, 0]
    return (a + th[:, None] / C).astype(np.float32)


def confidence_weighted(Z, split):
    P = stack3(Z, split); c = P.max(2)
    w = c / c.sum(1, keepdims=True)
    return np.einsum("nk,nkc->nc", w, P)


# ----------------------------------------------------------------------------- driver
def run(cfg: dict, ds: str, seed: int, force=False):
    rd = run_dir(cfg, ds, seed)
    if (rd / "fusion.json").exists() and not force:
        print(f"[skip] {rd}/fusion.json"); return
    Z = load_comp(rd)
    C = Z["p_cnn_val"].shape[1]
    benign = int(Z["benign"]) if int(Z["benign"]) >= 0 else None
    y = Z["y_test"]; fm = cfg["fpr_mode"]
    Pv, Pt = stack3(Z, "val"), stack3(Z, "test")
    yv = Z["y_val"]
    bud = cfg["budgets"]; K = cfg["k_regimes"]
    res, preds, params, post_keep = {}, {}, {}, {}
    sv = {n: Z[f"sig_{n}_val"] for n in H.SIGNAL_NAMES}; st = {n: Z[f"sig_{n}_test"] for n in H.SIGNAL_NAMES}
    rng = np.random.default_rng(stage_seed(seed, "shuffle"))
    t0 = time.time()

    def record(name, Pfused_test, meta=None, keep=False):
        preds[name] = Pfused_test.argmax(1).astype(np.int16)
        res[name] = {**evaluate(y, preds[name], C, benign, fm), **(meta or {})}
        if keep:
            post_keep[name] = Pfused_test

    def dwf(name, sig_v, sig_t, K_=K, free=True, fixed=None, keep=False):
        par = tune_dwf([{"P": Pv, "y": yv, "sig": sig_v, "C": C}], K_, seed, bud["dwf"], free, fixed, cfg["tune_rows"]["fusion"], name)
        params[name] = par
        record(name, apply_dwf(par, Pt, sig_t), {"val_macro_f1": par["val_macro_f1"], "trials": par["trials"], "K": K_}, keep)
        return par

    # reference: equal weights
    record("equal", Pt.mean(1), {"trials": 0}, keep=True)
    # DWF (Shannon, 10 bins) -- the proposed method
    dw = dwf("dwf", sv["shannon10"], st["shannon10"], keep=True)
    # A -- controls
    fx = dwf("A1_fixed", np.zeros_like(sv["shannon10"]), np.zeros_like(st["shannon10"]), K_=1, keep=True)
    freq = np.bincount(regime_of(sv["shannon10"], np.array(dw["bounds"])), minlength=K) / len(yv)
    rv = rng.choice(K, size=len(yv), p=freq).astype(float); rt = rng.choice(K, size=len(y), p=freq).astype(float)
    dwf("A2_random_regime", rv, rt, K, False, np.arange(K - 1) + 0.5)
    dwf("A3_shuffled_entropy", rng.permutation(sv["shannon10"]), rng.permutation(st["shannon10"]))
    dwf("A4_var_pkt", sv["var_pkt"], st["var_pkt"])
    dwf("A5_constant", np.zeros(len(yv)), np.zeros(len(y)), K, False, np.arange(K - 1) + 0.5)
    # B -- alternative measures / bins
    dwf("B1_renyi2", sv["renyi2"], st["renyi2"]); dwf("B2_tsallis2", sv["tsallis2"], st["tsallis2"])
    dwf("B3_shannon5", sv["shannon5"], st["shannon5"]); dwf("B4_shannon20", sv["shannon20"], st["shannon20"])
    dwf("B5_shannon50", sv["shannon50"], st["shannon50"])
    dwf("B6_predictive", H.predictive_entropy([Pv[:, i] for i in range(3)]), H.predictive_entropy([Pt[:, i] for i in range(3)]))
    # C -- regime structure
    qb = np.percentile(Z["sigfit_shannon10"], 100 * np.arange(1, K) / K)
    dwf("C1_quantile_bounds", sv["shannon10"], st["shannon10"], K, False, qb)
    bestK, bestv = None, -1
    for k in (2, 3, 7):
        par = dwf(f"C2_K{k}", sv["shannon10"], st["shannon10"], k)
        if par["val_macro_f1"] > bestv:
            bestK, bestv = k, par["val_macro_f1"]
    res["C2_selected"] = {**res[f"C2_K{bestK}"], "selected_K": bestK}
    preds["C2_selected"] = preds[f"C2_K{bestK}"]
    # D -- learned fusion
    P, meta = stacking(Z, cfg, seed, bud["stack"]); record("D1_stacking", P, meta)
    P, meta = learned_gate(Z, cfg, seed, bud["gate"], False); record("D2_gate_H", P, meta)
    P, meta = learned_gate(Z, cfg, seed, bud["gate"], True); record("D3_gate_H_HFR", P, meta)
    if "p_moe_test" in Z:
        record("D4_moe", Z["p_moe_test"], {"trials": 0})
    alphas = [float((Z[f"p_{c}_val"].argmax(1) == yv).mean()) for c in COMP3]
    record("D5_dempster_shafer", dempster_shafer(Z, "test", alphas), {"alphas": alphas, "trials": 0})
    record("D6_confidence_weighted", confidence_weighted(Z, "test"), {"trials": 0})
    # E -- competitors on identical inputs
    for e, nm in (("e1", "E1_cnn_attention"), ("e2", "E2_bilstm_rf"), ("e3", "E3_tuned_gbdt")):
        if f"p_{e}_test" in Z:
            record(nm, Z[f"p_{e}_test"], {"trials": 0})
    # standalone components (for Table 16 / sanity)
    for i, c in enumerate(COMP3):
        record(f"comp_{c}", Pt[:, i], {"trials": 0})

    # per-regime breakdown on test with the tuned DWF boundaries (Table 16)
    b = np.array(dw["bounds"]); r = regime_of(st["shannon10"], b)
    reg = []
    for k in range(K):
        m = r == k
        row = {"regime": k, "share": float(m.mean()), "n": int(m.sum())}
        if m.sum():
            for nm, pr in (("cnn", Pt[:, 0].argmax(1)), ("bilstm", Pt[:, 1].argmax(1)), ("gbdt", Pt[:, 2].argmax(1)), ("dwf", preds["dwf"])):
                row[nm] = evaluate(y[m], pr[m], C, benign, fm)["macro_f1"]
        reg.append(row)

    # AUC and posterior sub-sample for DeLong
    from sklearn.metrics import roc_auc_score
    dl = subsample(len(y), 100_000, stage_seed(seed, "bootstrap"))
    auc = {}
    for nm in ("dwf", "A1_fixed", "equal"):
        yy, PP = y[dl], post_keep[nm][dl]
        aucs = []
        for c in range(C):
            pos = yy == c
            if 0 < pos.sum() < len(yy):
                aucs.append(roc_auc_score(pos, PP[:, c]))
        auc[nm] = float(np.mean(aucs))
    for nm in ("dwf", "A1_fixed"):
        res[nm]["auc_roc"] = auc[nm]
    res["equal"]["auc_roc"] = auc["equal"]

    np.savez_compressed(rd / "fusion_preds.npz", **preds, y_test=y, block_test=Z["block_test"], delong_idx=dl,
                        delong_dwf=post_keep["dwf"][dl].astype(np.float16), delong_fixed=post_keep["A1_fixed"][dl].astype(np.float16))
    write_json(rd / "dwf_params.json", {k: v for k, v in params.items()})
    write_json(rd / "fusion.json", {"results": res, "regimes": reg, "wall_seconds": time.time() - t0})
    print(f"[done] fusion {ds} seed {seed}: dwf F1={res['dwf']['macro_f1']:.3f} fixed={res['A1_fixed']['macro_f1']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(load_cfg(a.cfg), a.dataset, a.seed, a.force)


if __name__ == "__main__":
    main()
