"""Experiments behind Tables 21-23 (resampling sensitivity, attribution stability, feature-harmonisation / WSN controls).

    python -m hai_dwf.extra --cfg config/smoke_ext.yaml --out results_ext        # needs the runs produced by run_ext.sh / run_all.py

Table 21: reads runs_t21_<strategy>/edge_rare/seed*/ (components + fusion with cfg['resampling'] in none|ros|smote|borderline).
Table 22: Integrated Gradients (CNN, BiLSTM; own TF implementation, midpoint rule, 64 steps), TreeSHAP (LightGBM pred_contrib),
          entropy-permutation importance of the DWF gate, regime-weighted ensemble attribution.
Table 23: raw-only / no-WSN HFR controls (runs_t23_*), Tier-1 GBDT (int8), simulated LEACH rotation applied to the test traffic.
Everything is reported as mean +- SD over the seeds in cfg['seeds']; intervals are Student-t.
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, t as tdist

from . import data as D, entropy as H, fusion as Fu
from .common import load_cfg, run_dir, read_json, evaluate, stage_seed, write_json

RARE = ("Slowloris", "Botnet-ARES")


def ms(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def ci(x, lvl=0.95):
    x = np.asarray(x, float); m, s = ms(x)
    h = tdist.ppf(0.5 + lvl / 2, len(x) - 1) * s / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, m - h, m + h


def f1_class(y, p, c):
    tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
    return 100 * 2 * tp / max(2 * tp + fp + fn, 1)


# =============================================================================== Table 21
def table21(cfg, seeds):
    names = {"none": "Class weights only (no resampling)", "ros": "Random oversampling", "smote": "SMOTE (proposed)", "borderline": "Borderline-SMOTE"}
    out, P = {}, []
    for st, label in names.items():
        c = dict(cfg); c["out_dir"] = f"runs_t21_{st}"
        mf, fpr, r1, r2 = [], [], [], []
        for s in seeds:
            rd = run_dir(c, "edge_rare", s)
            fz = np.load(rd / "fusion_preds.npz"); z = np.load(rd / "components.npz", allow_pickle=True)
            cl = [str(x) for x in z["classes"]]; y, p = fz["y_test"], fz["dwf"].astype(int)
            res = read_json(rd / "fusion.json")["results"]["dwf"]
            mf.append(res["macro_f1"]); fpr.append(res["fpr"])
            r1.append(f1_class(y, p, cl.index(RARE[0]))); r2.append(f1_class(y, p, cl.index(RARE[1])))
        out[st] = dict(macro_f1=ms(mf), fpr=ms(fpr), r1=ci(r1), r2=ci(r2), n_seeds=len(seeds))
        m, sd = ms(mf)
        P += [dict(table="T21", row=label, col=1, text=f"{m:.2f} ± {sd:.2f}"),
              dict(table="T21", row=label, col=2, text="{:.1f} [{:.1f}, {:.1f}]".format(*[min(max(v, 0), 100) for v in ci(r1)])),
              dict(table="T21", row=label, col=3, text="{:.1f} [{:.1f}, {:.1f}]".format(*[min(max(v, 0), 100) for v in ci(r2)])),
              dict(table="T21", row=label, col=4, text="{:.2f} ± {:.2f}".format(*ms(fpr)))]
    P.append(dict(table="T21", note=f"DEMONSTRATION values from SYNTHETIC data (N = {len(seeds)} seeds; mean ± SD for macro-F1/FPR, mean [95% Student-t interval] for class F1; "
                                    "the two rare classes of the synthetic set carry the Table-21 names). Not experimental results."))
    return out, P


# =============================================================================== Table 22
def _tf():
    import tensorflow as tf
    return tf


def ig(model, X, base, cls, C, steps=64, chunk=125):
    """Integrated Gradients of p_cls (midpoint rule). Returns attributions (same shape as X) and the completeness residual."""
    tf = _tf(); out = np.zeros_like(X); res = np.zeros(len(X)); delta = np.zeros(len(X))
    al = (np.arange(steps) + 0.5) / steps
    for i in range(0, len(X), chunk):
        x = tf.constant(X[i:i + chunk]); b = tf.constant(base[i:i + chunk]); oh = tf.one_hot(cls[i:i + chunk], C)
        tot = tf.zeros_like(x)
        for a in al:
            xi = b + a * (x - b)
            with tf.GradientTape() as g:
                g.watch(xi); sel = tf.reduce_sum(model(xi, training=False) * oh, 1)
            tot += g.gradient(sel, xi)
        at = ((x - b) * tot / steps).numpy(); out[i:i + chunk] = at
        fx = tf.reduce_sum(model(x, training=False) * oh, 1).numpy(); fb = tf.reduce_sum(model(b, training=False) * oh, 1).numpy()
        delta[i:i + chunk] = fx - fb; res[i:i + chunk] = at.reshape(len(at), -1).sum(1) - (fx - fb)
    return out, 100 * np.abs(res).sum() / max(np.abs(delta).sum(), 1e-9)


def share(a):
    g = np.abs(a).mean(0); return 100 * g / max(g.sum(), 1e-12)


def tau(a, b):
    return float(kendalltau(a, b)[0])


def table22(cfg, seeds, ds="ciciot2023"):
    from .adversarial import Ctx
    rec = []
    for s in seeds:
        c = Ctx(cfg, ds, s); W = c.W0; y = c.y; C = c.C; n = len(W)
        rng = np.random.default_rng(stage_seed(s, "attribution"))
        pc = c.cnn.predict(W[:, -1, :], verbose=0); pb = c.bil.predict(W, verbose=0)
        kc, kb = pc.argmax(1), pb.argmax(1)
        bases = {"zero": np.zeros_like(W),
                 "benign_mean": np.broadcast_to(W[y == 0].mean((0, 1)) if (y == 0).any() else W.mean((0, 1)), W.shape).astype(np.float32).copy(),
                 "noise": np.clip(rng.normal(0, 0.1, W.shape), -1, 1).astype(np.float32)}
        r = dict(seed=s, names=c.names, n=int(n))
        Ac, Ab = {}, {}
        for bn, B in bases.items():
            a_c, ce_c = ig(c.cnn, W[:, -1, :], B[:, -1, :], kc, C)
            a_b, ce_b = ig(c.bil, W, B, kb, C)
            Ac[bn], Ab[bn] = a_c, a_b.sum(1)               # BiLSTM: summed over T
            if bn == "zero":
                r["ce_cnn"], r["ce_bil"] = ce_c, ce_b
        r["share_cnn"], r["share_bil"] = share(Ac["zero"]), share(Ab["zero"])
        r["bs_cnn"] = float(np.mean([tau(share(Ac["zero"]), share(Ac[k])) for k in ("benign_mean", "noise")]))
        r["bs_bil"] = float(np.mean([tau(share(Ab["zero"]), share(Ab[k])) for k in ("benign_mean", "noise")]))
        # TreeSHAP (LightGBM, log-odds of the predicted class)
        pcn = c.gb.predict(W[:, -1, :], pred_contrib=True); F = W.shape[2]
        kg = c.gb.predict(W[:, -1, :]).argmax(1)
        sh = pcn.reshape(n, C, F + 1)[np.arange(n), kg, :F]
        r["share_gb"] = share(sh)
        # DWF gate: shuffle H(x) across samples
        Pst = c.posteriors(W); sig = H.shannon(c.pp.unit(W), cfg["bins"])
        base_f1 = evaluate(y, Fu.apply_dwf(c.dwf, Pst, sig).argmax(1), C, None, cfg["fpr_mode"])["macro_f1"]
        d = [evaluate(y, Fu.apply_dwf(c.dwf, Pst, rng.permutation(sig)).argmax(1), C, None, cfg["fpr_mode"])["macro_f1"] - base_f1 for _ in range(20)]
        r["gate_delta"] = float(np.mean(d)); r["gate_W"] = np.asarray(c.dwf["W"]).ravel().tolist()
        # regime-weighted ensemble attribution (per-sample L1-normalised component attributions)
        Wr = np.array(c.dwf["W"])[Fu.regime_of(sig, np.array(c.dwf["bounds"]))]               # (n,3)

        def l1(a):
            a = np.abs(a); return a / np.maximum(a.sum(1, keepdims=True), 1e-12)
        ens = lambda k: (Wr[:, 0:1] * l1(Ac[k]) + Wr[:, 1:2] * l1(Ab[k]) + Wr[:, 2:3] * l1(sh)).mean(0)
        e0 = ens("zero"); r["share_ens"] = (100 * e0 / e0.sum()).tolist()
        r["bs_ens"] = float(np.mean([tau(e0, ens(k)) for k in ("benign_mean", "noise")]))
        for k in ("share_cnn", "share_bil", "share_gb"):
            r[k] = r[k].tolist()
        rec.append(r); print("attribution seed", s, "done", flush=True)
    return rec


def _top3(rec, key):
    names = sorted({n for r in rec for n in r["names"]})
    M = np.array([[dict(zip(r["names"], r[key])).get(n, 0.0) for n in names] for r in rec])
    mu, sd = M.mean(0), M.std(0, ddof=1) if len(rec) > 1 else np.zeros(M.shape[1])
    top = np.argsort(-mu)[:3]
    return "; ".join(f"{names[i]} ({mu[i]:.1f} ± {sd[i]:.1f})" for i in top)


def _stab(rec, key):
    taus = []
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            a, b = dict(zip(rec[i]["names"], rec[i][key])), dict(zip(rec[j]["names"], rec[j][key]))
            com = sorted(set(a) & set(b)); taus.append(tau([a[k] for k in com], [b[k] for k in com]))
    return ms(taus)


def table22_patches(rec):
    P, f = [], lambda v: "{:.2f} ± {:.2f}".format(*v)
    row = {"cnn": "CNN (Integrated Gradients)", "bil": "BiLSTM (Integrated Gradients, summed over T)", "gb": "LightGBM (TreeSHAP)",
           "ens": "Ensemble (regime-weighted sum)"}
    for k, label in row.items():
        P += [dict(table="T22", row=label, col=1, text=_top3(rec, f"share_{k}")), dict(table="T22", row=label, col=2, text=f(_stab(rec, f"share_{k}")))]
    P += [dict(table="T22", row=row["cnn"], col=3, text=f(ms([r["bs_cnn"] for r in rec]))), dict(table="T22", row=row["cnn"], col=4, text=f(ms([r["ce_cnn"] for r in rec]))),
          dict(table="T22", row=row["bil"], col=3, text=f(ms([r["bs_bil"] for r in rec]))), dict(table="T22", row=row["bil"], col=4, text=f(ms([r["ce_bil"] for r in rec]))),
          dict(table="T22", row=row["ens"], col=3, text=f(ms([r["bs_ens"] for r in rec])))]
    g = "DWF gate: H(x) permutation"
    gw = [dict(zip(range(len(r["gate_W"])), r["gate_W"])) for r in rec]
    gt = []
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            gt.append(tau(rec[i]["gate_W"], rec[j]["gate_W"]))
    P += [dict(table="T22", row=g, col=1, text="Δ macro-F1 = {:+.2f} ± {:.2f} pp".format(*ms([r["gate_delta"] for r in rec]))),
          dict(table="T22", row=g, col=2, text=f(ms(gt)))]
    P.append(dict(table="T22", note=f"DEMONSTRATION values from SYNTHETIC data (CIC-IoT-like smoke set, N = {len(rec)} seeds, {rec[0]['n']} test windows per seed; mean ± SD). "
                                    "IG: midpoint rule, 64 steps, baselines = zero / benign mean / noise; completeness error = Σ|ΣIG − (F(x) − F(x0))| / Σ|F(x) − F(x0)|. "
                                    "Gate row: stability = Kendall τ of the regime-weight profiles across seeds. Not experimental results."))
    return P


# =============================================================================== Table 23
def _load_part(cfg, ds, seed):
    rd = run_dir(cfg, ds, seed)
    with open(rd / "preproc.pkl", "rb") as f:
        o = pickle.load(f)
    df = D.load_raw(cfg, ds); P = np.load(rd / "partitions.npz")
    return rd, o, df, P


def tier1(cfg, seed, ds="wsn_ds"):
    import lightgbm as lgb
    rd, o, df, P = _load_part(cfg, ds, seed); pp, spec = o["pp"], o["spec"]
    Z = pp.transform(D.apply_hfr(df, spec)); q = np.clip(np.round(Z * 127), -127, 127).astype(np.int8).astype(np.float32)   # int8 inputs
    part, y, blk = P["part"], P["y"], P["block"]; C = int(y.max()) + 1
    fit, te = np.flatnonzero(part == 0), np.flatnonzero(part == 2)
    cnt = np.bincount(y[fit], minlength=C); w = (len(fit) / (C * np.maximum(cnt, 1)))[y[fit]]
    prm = dict(objective="multiclass", num_class=C, num_leaves=15, max_depth=4, learning_rate=0.2, min_data_in_leaf=10, verbose=-1,
               seed=stage_seed(seed, "tier1"), deterministic=True, force_row_wise=True)
    b = lgb.train(prm, lgb.Dataset(q[fit], y[fit], weight=w), num_boost_round=30)
    pr = b.predict(q[te]).argmax(1)
    benign = int(np.load(rd / "components.npz", allow_pickle=True)["benign"]); benign = benign if benign >= 0 else None
    nodes = len(b.trees_to_dataframe())
    return evaluate(y[te], pr, C, benign, cfg["fpr_mode"]), nodes, C


def leach(cfg, seed, ds="wsn_ds", round_len=50, n_nodes=20):
    from .adversarial import Ctx
    c = Ctx(cfg, ds, seed); rd, o, df, P = _load_part(cfg, ds, seed)
    rows = np.flatnonzero(P["part"] == 2); d = df.iloc[rows].copy(); rng = np.random.default_rng(stage_seed(seed, "leach"))
    rnd = np.arange(len(d)) // round_len                                              # one LEACH round = round_len records
    node = rng.integers(0, n_nodes, rnd.max() + 1); ch = rng.random(rnd.max() + 1) < 0.1           # ~10 % cluster heads per round
    off_lqi = rng.normal(0, 1, n_nodes); hops = rng.integers(1, 5, n_nodes)
    for col in d.columns:
        lc = col.lower()
        if "lqi" in lc:
            d[col] = d[col].to_numpy(float) + off_lqi[node[rnd]] * np.nanstd(df[col])
        elif "hop" in lc:
            d[col] = np.where(ch[rnd], 1, hops[node[rnd]])
        elif "rout" in lc:
            d[col] = d[col].to_numpy(float) + (np.arange(len(d)) % round_len == 0) * rng.integers(1, 4, len(d))
        elif lc in ("pkts_count", "bytes_per_s"):
            d[col] = d[col].to_numpy(float) * np.where(ch[rnd], 2.5, 1.0)              # cluster heads aggregate and forward
    Z = c.pp.transform(D.apply_hfr(d, c.spec)); idx = D.window_index(P["block"][rows], cfg["window"])
    W = Z[idx].astype(np.float32); y = P["y"][rows]
    pred = np.concatenate([c.predict(W[i:i + 2000]).argmax(1) for i in range(0, len(W), 2000)])
    benign = int(np.load(run_dir(cfg, ds, seed) / "components.npz", allow_pickle=True)["benign"]); benign = benign if benign >= 0 else None
    return evaluate(y, pred, c.C, benign, cfg["fpr_mode"])


def table23(cfg, seeds):
    full = lambda ds, s, d="runs_smoke": read_json(Path(d) / ds / f"seed{s}" / "fusion.json")["results"]["dwf"]
    rows, P = {}, []
    spec = [("Raw-only HFR (no engineered components)", "NF-ToN-IoT-v2", "nf_ton_iot_v2", "runs_t23_raw"),
            ("Raw-only HFR (no engineered components)", "WSN-DS", "wsn_ds", "runs_t23_raw"),
            ("Without WSN-specific components (LQI, hop count, routing changes)", "WSN-DS", "wsn_ds", "runs_t23_nowsn")]
    for label, dname, ds, d in spec:
        m, f, dl = [], [], []
        for s in seeds:
            r = read_json(Path(d) / ds / f"seed{s}" / "fusion.json")["results"]["dwf"]; ref = full(ds, s)
            m.append(r["macro_f1"]); f.append(r["fpr"]); dl.append(r["macro_f1"] - ref["macro_f1"])
        rows[f"{label}|{dname}"] = dict(m=ms(m), f=ms(f), d=ms(dl))
    tm, tf_, td, nodes = [], [], [], []
    lm, lf, ld = [], [], []
    for s in seeds:
        r, nd, C = tier1(cfg, s); ref = full("wsn_ds", s)
        tm.append(r["macro_f1"]); tf_.append(r["fpr"]); td.append(r["macro_f1"] - ref["macro_f1"]); nodes.append(nd)
        r = leach(cfg, s); lm.append(r["macro_f1"]); lf.append(r["fpr"]); ld.append(r["macro_f1"] - ref["macro_f1"])
    rows["Tier 1|WSN-DS"] = dict(m=ms(tm), f=ms(tf_), d=ms(td), nodes=float(np.mean(nodes)))
    rows["LEACH|WSN-DS"] = dict(m=ms(lm), f=ms(lf), d=ms(ld))
    keymap = {f"{a}|{b}": f"{a}|{b}" for a, b, _, _ in spec}
    keymap["Tier 1|WSN-DS"] = "Tier 1 constrained (GBDT only, int8, ≤ 64 KB SRAM)|WSN-DS"
    keymap["LEACH|WSN-DS"] = "Simulated LEACH traffic with node rotation|WSN-DS"
    for k, v in rows.items():
        key = keymap[k]
        P += [dict(table="T23", row=key, col=2, text="{:.2f} ± {:.2f}".format(*v["m"])), dict(table="T23", row=key, col=3, text="{:.2f} ± {:.2f}".format(*v["f"])),
              dict(table="T23", row=key, col=4, text="{:+.2f} ± {:.2f}".format(*v["d"]))]
    kb = rows["Tier 1|WSN-DS"]["nodes"] * 6 / 1024
    P.append(dict(table="T23", row=keymap["Tier 1|WSN-DS"], col=5,
                  text=f"≈ {kb:.1f} KB flash, ≈ 0.6 KB SRAM (estimated from node count, 6 B/node; not measured on a board)"))
    P.append(dict(table="T23", note=f"DEMONSTRATION values from SYNTHETIC data (N = {len(seeds)} seeds; mean ± SD; Δ = paired difference to the full HAI+DWF on the same seed). "
                                    "Tier 1: LightGBM, 30 rounds/class, 15 leaves, depth 4, int8 inputs. LEACH: 50-record rounds, 20 nodes, ~10 % cluster heads per round. Not experimental results."))
    return rows, P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default="config/smoke_ext.yaml"); ap.add_argument("--out", default="results_ext")
    ap.add_argument("--only", nargs="+", default=["t21", "t22", "t23"])
    a = ap.parse_args(); cfg = load_cfg(a.cfg); seeds = cfg["seeds"]; out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    patches, summ = [], {}
    if "t21" in a.only:
        s, p = table21(cfg, seeds); patches += p; summ["t21"] = s
    if "t22" in a.only:
        cfg['adversarial']['n_eval'] = 250
        rec = table22(cfg, seeds); summ["t22"] = rec; patches += table22_patches(rec)
    if "t23" in a.only:
        s, p = table23(cfg, seeds); patches += p; summ["t23"] = s
    write_json(out / "patches_ext.json", patches); write_json(out / "summary_ext.json", summ)
    for p in patches:
        if "text" in p:
            print(p["table"], "|", p["row"][:40], "|", p["col"], "|", p["text"])


if __name__ == "__main__":
    main()
