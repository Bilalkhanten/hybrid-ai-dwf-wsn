"""Stage 1: for one (dataset, master seed) train the three components (+ E1-E3 competitors, D4 MoE), and store posteriors,
control signals and partition metadata.  Stage 2 (fusion.py) consumes ONLY the stored posteriors, so every fusion variant
sees identical component outputs (Section 3.9).

    python -m hai_dwf.components --cfg config/default.yaml --dataset ciciot2023 --seed 0
"""
from __future__ import annotations
import argparse, pickle, time
from pathlib import Path
import numpy as np

from .common import (load_cfg, seed_all, stage_seed, run_dir, write_json, software_versions, all_stage_seeds,
                     evaluate)
from . import data as D
from . import entropy as H
from . import models as M


def _pkt_component(spec, dcfg):
    import re
    names = [s["name"] for s in spec]
    if dcfg.get("packet_size_feature") in names:
        return names.index(dcfg["packet_size_feature"])
    for i, n in enumerate(names):
        if re.search(r"(len|size|bytes)", n, re.I):
            return i
    return 0


def balance(Z, idx, y, cfg, seed):
    """Class-capped random undersampling + resampling of the FITTING partition only.
    cfg['resampling'] in {smote (default), none (inverse-frequency class weights only), ros, borderline}. Returns Xw, yw, sample_weight|None."""
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
    strat = cfg.get("resampling", "smote")
    rng = np.random.default_rng(stage_seed(seed, "smote"))
    cap, k = cfg["smote"]["cap_per_class"], cfg["smote"]["k"]
    C = int(y.max()) + 1
    sel = []
    for c in range(C):
        ic = np.flatnonzero(y == c)
        if len(ic) > cap:
            ic = rng.choice(ic, cap, replace=False)
        sel.append(ic)
    sel = np.concatenate(sel)
    T, F = idx.shape[1], Z.shape[1]
    Xw = Z[idx[sel]].reshape(len(sel), -1); yw = y[sel]
    sw = None
    if strat == "none":
        cnt = np.bincount(yw, minlength=C).astype(float)
        sw = (len(yw) / (C * np.maximum(cnt, 1)))[yw].astype(np.float32)
    elif cfg["smote"]["enabled"]:
        cnt = np.bincount(yw, minlength=C)
        med = int(np.median(cnt[cnt > 0]))
        target = {c: min(cap, max(int(cnt[c]), med)) for c in range(C) if 0 < cnt[c] < min(cap, max(int(cnt[c]), med))}
        small = {c: t for c, t in target.items() if cnt[c] <= k or strat == "ros"}
        big = {c: t for c, t in target.items() if c not in small}
        if small:
            Xw, yw = RandomOverSampler(sampling_strategy=small, random_state=stage_seed(seed, "smote")).fit_resample(Xw, yw)
        if big:
            kk = max(1, min(k, min(int(cnt[c]) for c in big) - 1))
            Cls = BorderlineSMOTE if strat == "borderline" else SMOTE
            Xw, yw = Cls(sampling_strategy={c: t for c, t in big.items()}, k_neighbors=kk,
                         random_state=stage_seed(seed, "smote")).fit_resample(Xw, yw)
    return Xw.reshape(-1, T, F).astype(np.float32), yw.astype(np.int64), sw


def _mf1(y, p, C):
    return evaluate(y, p, C)["macro_f1"]


def tune_lgbm(Xtr, ytr, Xva, yva, C, cfg, seed, trials, sw=None):
    import lightgbm as lgb, optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rs = np.random.default_rng(stage_seed(seed, "tpe"))
    n_tune = cfg["tune_rows"]["lgbm"]
    it = rs.permutation(len(Xtr))[:n_tune]; iv = rs.permutation(len(Xva))[:n_tune]
    base = dict(objective="multiclass", num_class=C, verbosity=-1, deterministic=True, force_row_wise=True,
                seed=stage_seed(seed, "lgbm_bagging"), bagging_seed=stage_seed(seed, "lgbm_bagging"),
                feature_fraction_seed=stage_seed(seed, "lgbm_bagging"), num_threads=0)

    def sample(tr):
        return dict(num_leaves=tr.suggest_int("num_leaves", 15, 255), learning_rate=tr.suggest_float("lr", 0.01, 0.3, log=True),
                    min_child_samples=tr.suggest_int("mcs", 5, 100), subsample=tr.suggest_float("subsample", 0.5, 1.0),
                    subsample_freq=1, colsample_bytree=tr.suggest_float("colsample", 0.5, 1.0),
                    reg_alpha=tr.suggest_float("alpha", 1e-8, 10, log=True), reg_lambda=tr.suggest_float("lambda", 1e-8, 10, log=True))

    def obj(tr):
        p = {**base, **sample(tr)}
        b = lgb.train(p, lgb.Dataset(Xtr[it], ytr[it], weight=None if sw is None else sw[it]), num_boost_round=300,
                      valid_sets=[lgb.Dataset(Xva[iv], yva[iv])], callbacks=[lgb.early_stopping(20, verbose=False)])
        return _mf1(yva[iv], b.predict(Xva[iv]).argmax(1), C)

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=stage_seed(seed, "tpe")))
    st.optimize(obj, n_trials=trials)
    p = {**base, **sample(optuna.trial.FixedTrial(st.best_params))}
    b = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=sw), num_boost_round=1000, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    return b, st.best_params, trials


def tune_xgb_cat(Xtr, ytr, Xva, yva, C, cfg, seed, trials):
    """E3: tuned XGBoost and CatBoost (same budget each); the better one on validation is kept."""
    import optuna, xgboost as xgb
    from catboost import CatBoostClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rs = np.random.default_rng(stage_seed(seed, "tpe"))
    n = cfg["tune_rows"]["lgbm"]; it = rs.permutation(len(Xtr))[:n]; iv = rs.permutation(len(Xva))[:n]
    res = {}

    def run_xgb(tr):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=tr.suggest_int("d", 3, 12), learning_rate=tr.suggest_float("lr", 0.02, 0.3, log=True),
                              subsample=tr.suggest_float("ss", 0.5, 1), colsample_bytree=tr.suggest_float("cs", 0.5, 1),
                              reg_lambda=tr.suggest_float("l", 1e-6, 10, log=True), tree_method="hist", random_state=stage_seed(seed, "lgbm_bagging"),
                              n_jobs=-1)
        return m

    def run_cat(tr):
        return CatBoostClassifier(iterations=200, depth=tr.suggest_int("d", 3, 10), learning_rate=tr.suggest_float("lr", 0.02, 0.3, log=True),
                                  l2_leaf_reg=tr.suggest_float("l2", 1e-2, 10, log=True), loss_function="MultiClass", verbose=0,
                                  random_seed=stage_seed(seed, "lgbm_bagging"), thread_count=-1)

    def ytrfix(y):  # xgboost needs contiguous labels
        return y

    out = {}
    for name, mk in (("xgboost", run_xgb), ("catboost", run_cat)):
        def obj(tr, mk=mk):
            m = mk(tr)
            yy = ytr[it]
            if name == "xgboost":
                classes = np.unique(yy)
                if len(classes) < C:
                    return 0.0
            m.fit(Xtr[it], yy)
            return _mf1(yva[iv], np.asarray(m.predict(Xva[iv])).astype(int).ravel(), C)
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=stage_seed(seed, "tpe")))
        st.optimize(obj, n_trials=trials)
        m = mk(optuna.trial.FixedTrial(st.best_params)); m.fit(Xtr, ytr)
        P = np.zeros((len(Xva), C), np.float32)
        pv = m.predict_proba(Xva); P[:, np.unique(ytr)] = pv
        out[name] = (m, st.best_value, P)
    best = max(out, key=lambda k: out[k][1])
    return best, out[best][0], out


def run(cfg: dict, ds: str, seed: int, baselines: bool = True, max_rows: int | None = None, force: bool = False):
    import tensorflow as tf
    rd = run_dir(cfg, ds, seed)
    if (rd / "components.npz").exists() and not force:
        print(f"[skip] {rd}/components.npz exists"); return
    seed_all(stage_seed(seed, "init"))
    dcfg = cfg["datasets"][ds]; T = cfg["window"]; F = cfg["n_hfr"]
    df = D.load_raw(cfg, ds, max_rows)
    df[dcfg["label_col"]] = df[dcfg["label_col"]].astype(str)
    classes = sorted(df[dcfg["label_col"]].unique()); C = len(classes)
    benign = classes.index(str(dcfg["benign_label"])) if str(dcfg["benign_label"]) in classes else None
    P = D.make_partitions(df, dcfg, cfg, seed, classes)
    sha = D.save_partitions(P, rd)
    part, y_all, blk = P["part"], P["y"], P["block"]
    ov = cfg.get("hfr_overrides", {}).get(ds)
    spec = D.fit_hfr_spec(df[part == 0], dcfg, F, seed, ov, cfg.get("hfr_mode", "default"))
    write_json(rd / "hfr_spec.json", spec)
    Xraw = D.apply_hfr(df, spec)
    pp = D.Preproc(cfg["imputer_iters"], seed).fit(Xraw[part == 0])
    Z = pp.transform(Xraw); U = pp.unit(Z)
    pkt = _pkt_component(spec, dcfg)
    T0 = time.time()

    parts = {}
    for k, name in enumerate(["fit", "val", "test"]):
        rows = np.flatnonzero(part == k)
        parts[name] = dict(rows=rows, Z=Z[rows], U=U[rows], y=y_all[rows], block=blk[rows])
        parts[name]["idx"] = D.window_index(parts[name]["block"], T)

    def signals(pt, chunk=4000):
        acc = {}
        for sl in D.iter_chunks(len(pt["y"]), chunk):
            s = H.all_signals(pt["U"][pt["idx"][sl]], pkt, cfg["bins"])
            for kk, v in s.items():
                acc.setdefault(kk, []).append(v)
        return {kk: np.concatenate(v) for kk, v in acc.items()}

    sig = {n: signals(parts[n]) for n in parts}
    grid = np.arange(1, 100)
    sig_q = {kk: np.percentile(sig["fit"][kk], grid) for kk in H.SIGNAL_NAMES}

    # ---------------- training set (SMOTE on fitting partition only)
    Xw, yw, sw = balance(parts["fit"]["Z"], parts["fit"]["idx"], parts["fit"]["y"], cfg, seed)
    Xi = Xw[:, -1, :]
    # validation windows for early stopping (cap for memory)
    rs = np.random.default_rng(stage_seed(seed, "shuffle"))
    nv = len(parts["val"]["y"]); iv = np.sort(rs.permutation(nv)[: min(nv, cfg["tune_rows"]["fusion"])])
    Xvw = parts["val"]["Z"][parts["val"]["idx"][iv]]; yv = parts["val"]["y"][iv]
    Xvi = Xvw[:, -1, :]

    def seq(pt):
        return lambda sl: pt["Z"][pt["idx"][sl]]

    def inst(pt):
        return lambda sl: pt["Z"][pt["idx"][sl, -1]]

    post = {}; timing = {}; extra = {}

    def predict_all(model, kind):
        out = {}
        for n in ("val", "test"):
            fn = seq(parts[n]) if kind == "seq" else inst(parts[n])
            out[n] = M.predict_batched(model, fn, len(parts[n]["y"]))
        return out

    # CNN
    t = time.time(); seed_all(stage_seed(seed, "init"))
    cnn = M.build_cnn(C, F); M.compile_fit(cnn, Xi, yw, Xvi, yv, cfg["nn"], sw=sw); timing["cnn"] = time.time() - t
    post["cnn"] = predict_all(cnn, "inst"); cnn.save(rd / "cnn.keras")
    # BiLSTM
    t = time.time(); seed_all(stage_seed(seed, "init") + 1)
    bil = M.build_bilstm(C, T, F); M.compile_fit(bil, Xw, yw, Xvw, yv, cfg["nn"], sw=sw); timing["bilstm"] = time.time() - t
    post["bilstm"] = predict_all(bil, "seq"); bil.save(rd / "bilstm.keras")
    # LightGBM
    t = time.time()
    gb, gb_params, ntr = tune_lgbm(Xi, yw, Xvi, yv, C, cfg, seed, cfg["budgets"]["lgbm"], sw); timing["gbdt"] = time.time() - t
    post["gbdt"] = {n: gb.predict(parts[n]["Z"][parts[n]["idx"][:, -1]]).astype(np.float32) for n in ("val", "test")}
    gb.save_model(str(rd / "lgbm.txt"))
    extra["lgbm"] = {"best_params": gb_params, "trials": ntr}

    if baselines:
        # E1 CNN + attention
        t = time.time(); seed_all(stage_seed(seed, "init") + 2)
        e1 = M.build_cnn_attention(C, T, F); M.compile_fit(e1, Xw, yw, Xvw, yv, cfg["nn"]); timing["e1"] = time.time() - t
        post["e1"] = predict_all(e1, "seq")
        # E2 BiLSTM + RF
        from sklearn.ensemble import RandomForestClassifier
        t = time.time(); seed_all(stage_seed(seed, "init") + 3)
        e2n = M.build_bilstm_embed(C, T, F); M.compile_fit(e2n, Xw, yw, Xvw, yv, cfg["nn"])
        emb = tf.keras.Model(e2n.input, e2n.get_layer("emb").output)
        Etr = emb.predict(Xw, batch_size=4096, verbose=0)
        rf = RandomForestClassifier(200, n_jobs=-1, random_state=stage_seed(seed, "lgbm_bagging")).fit(Etr, yw)
        post["e2"] = {}
        for n in ("val", "test"):
            E = M.predict_batched(emb, seq(parts[n]), len(parts[n]["y"]))
            pr = np.zeros((len(E), C), np.float32); pr[:, rf.classes_] = rf.predict_proba(E); post["e2"][n] = pr
        timing["e2"] = time.time() - t
        # E3 XGBoost / CatBoost
        t = time.time()
        best, m3, allm = tune_xgb_cat(Xi, yw, Xvi, yv, C, cfg, seed, cfg["budgets"]["xgb_cat"])
        post["e3"] = {}
        for n in ("val", "test"):
            pr = np.zeros((len(parts[n]["y"]), C), np.float32)
            pr[:, np.unique(yw)] = m3.predict_proba(parts[n]["Z"][parts[n]["idx"][:, -1]])
            post["e3"][n] = pr
        timing["e3"] = time.time() - t; extra["e3_choice"] = {"selected": best, "val_macro_f1": {k: v[1] for k, v in allm.items()}}
        # D4 mixture of experts (jointly trained)
        t = time.time(); seed_all(stage_seed(seed, "init") + 4)
        moe = M.build_moe(C, T, F)
        moe.compile(tf.keras.optimizers.Adam(cfg["nn"]["lr"]), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
        moe.fit(Xw, yw, validation_data=(Xvw, yv), epochs=cfg["nn"]["epochs"], batch_size=cfg["nn"]["batch"], verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=cfg["nn"]["patience"], restore_best_weights=True)])
        post["moe"] = predict_all(moe, "seq"); timing["moe"] = time.time() - t

    # ---------------- persist
    arrs = {"classes": np.array(classes), "benign": np.array(-1 if benign is None else benign),
            "y_val": parts["val"]["y"], "y_test": parts["test"]["y"], "block_val": parts["val"]["block"],
            "block_test": parts["test"]["block"], "hfr_val": parts["val"]["Z"][parts["val"]["idx"][:, -1]],
            "hfr_test": parts["test"]["Z"][parts["test"]["idx"][:, -1]]}
    for n in ("val", "test"):
        for comp, d in post.items():
            arrs[f"p_{comp}_{n}"] = d[n]
        for kk, v in sig[n].items():
            arrs[f"sig_{kk}_{n}"] = v
    for kk, v in sig_q.items():
        arrs[f"sigq_{kk}"] = v
    for kk, v in sig["fit"].items():
        arrs[f"sigfit_{kk}"] = v[:: max(1, len(v) // 100000)]
    np.savez_compressed(rd / "components.npz", **arrs)
    with open(rd / "preproc.pkl", "wb") as f:
        pickle.dump({"pp": pp, "spec": spec, "pkt_comp": pkt}, f)
    comp_metrics = {c: evaluate(parts["test"]["y"], post[c]["test"].argmax(1), C, benign, cfg["fpr_mode"]) for c in post}
    write_json(rd / "manifest.json", {
        "dataset": ds, "seed": seed, "stage_seeds": all_stage_seeds(seed), "partition_sha256": sha,
        "n_classes": C, "classes": classes, "train_seconds": timing, "component_test_metrics": comp_metrics, "extra": extra,
        "balanced_train_rows": int(len(yw)), "versions": software_versions(), "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "wall_seconds": time.time() - T0})
    print(f"[done] {ds} seed {seed}: " + ", ".join(f"{c}={m['macro_f1']:.2f}" for c, m in comp_metrics.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(load_cfg(a.cfg), a.dataset, a.seed, not a.no_baselines, a.max_rows, a.force)


if __name__ == "__main__":
    main()
