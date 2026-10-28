"""Aggregate all runs into table patches for the manuscript + a plain-language results summary.

    python -m hai_dwf.aggregate --cfg config/default.yaml --deploy deploy/measurements.json --repo-url <URL> --out results

Writes  results/patches.json  (consumed by fill_docx.py)  and  results/results_summary.md  (decision-rule verdict, warnings,
cells that could not be filled and why).  Every number is computed from run files; a cell with no underlying run is NOT
invented -- it is listed under 'unfilled' in the summary and stays TBR/TBC in the document.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

from .common import load_cfg, read_json, write_json, run_dir, holm, all_stage_seeds
from . import stats as S

T15_ROWS = [("Reference: equal-weight fusion", "equal"), ("Reference: DWF, Shannon entropy (proposed)", "dwf"),
            ("A1.", "A1_fixed"), ("A2.", "A2_random_regime"), ("A3.", "A3_shuffled_entropy"), ("A4.", "A4_var_pkt"), ("A5.", "A5_constant"),
            ("B1.", "B1_renyi2"), ("B2.", "B2_tsallis2"), ("B3.", "B3_shannon5"), ("B4.", "B4_shannon20"), ("B5.", "B5_shannon50"),
            ("B6.", "B6_predictive"), ("C1.", "C1_quantile_bounds"), ("C2.", "C2_selected"),
            ("D1.", "D1_stacking"), ("D2.", "D2_gate_H"), ("D3.", "D3_gate_H_HFR"), ("D4.", "D4_moe"), ("D5.", "D5_dempster_shafer"),
            ("D6.", "D6_confidence_weighted"), ("E1.", "E1_cnn_attention"), ("E2.", "E2_bilstm_rf"), ("E3.", "E3_tuned_gbdt")]
FAMILIES = {"ref": ["equal"], "A": ["A1_fixed", "A2_random_regime", "A3_shuffled_entropy", "A4_var_pkt", "A5_constant"],
            "B": ["B1_renyi2", "B2_tsallis2", "B3_shannon5", "B4_shannon20", "B5_shannon50", "B6_predictive"],
            "C": ["C1_quantile_bounds", "C2_selected"], "D": ["D1_stacking", "D2_gate_H", "D3_gate_H_HFR", "D4_moe", "D5_dempster_shafer", "D6_confidence_weighted"],
            "E": ["E1_cnn_attention", "E2_bilstm_rf", "E3_tuned_gbdt"]}
HYP = {0: "gbdt", 1: "gbdt", 2: "bilstm", 3: "cnn", 4: "cnn"}


def fmt(x, nd=2): return f"{x:.{nd}f}"
def ms(v, nd=2): v = np.asarray(v, float); return f"{v.mean():.{nd}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}"


def load(cfg):
    R = {}
    for ds in cfg["dataset_order"]:
        R[ds] = {}
        for s in cfg["seeds"]:
            rd = run_dir(cfg, ds, s)
            if (rd / "fusion.json").exists():
                R[ds][s] = {"fusion": read_json(rd / "fusion.json"), "manifest": read_json(rd / "manifest.json") if (rd / "manifest.json").exists() else None,
                            "transfer": read_json(rd / "transfer.json") if (rd / "transfer.json").exists() else None,
                            "adv": read_json(rd / "adversarial.json") if (rd / "adversarial.json").exists() else None,
                            "dir": rd}
    return R


def vals(R, ds, key, metric):
    return np.array([R[ds][s]["fusion"]["results"][key][metric] for s in sorted(R[ds]) if key in R[ds][s]["fusion"]["results"]])


STG = ["capture", "hfr", "scale", "cnn", "bilstm", "lgbm", "dwf"]


def build(cfg, deploy_json=None, repo_url=None):
    R = load(cfg); order = cfg["dataset_order"]; src = order[0]
    patches, unfilled, notes, summ = [], [], [], []
    def P(table, row, col, text): patches.append({"table": table, "row": row, "col": col, "text": text})

    for ds in order:
        n = len(R[ds])
        if n < len(cfg["seeds"]):
            summ.append(f"WARNING: {ds}: {n}/{len(cfg['seeds'])} seeds completed")

    # ---- budgets check
    bud = {}
    for ds in order:
        for s, r in R[ds].items():
            for k, v in r["fusion"]["results"].items():
                if isinstance(v, dict) and v.get("trials"):
                    bud.setdefault(k, set()).add(v["trials"])
    bset = set().union(*[v for k, v in bud.items() if k in FAMILIES["A"] + FAMILIES["B"] + FAMILIES["C"][:1] + ["dwf"]]) if bud else set()
    if len(bset) > 1:
        summ.append(f"WARNING: tuning budgets differ across variants {sorted(bset)} -- the equal-budget statement of Section 3.9 does not hold for this run")

    # ================= Table 15
    if src in R and R[src]:
        seeds = sorted(R[src])
        dwf_f1 = vals(R, src, "dwf", "macro_f1"); fix_f1 = vals(R, src, "A1_fixed", "macro_f1")
        pv, fam_of = {}, {}
        for fam, keys in FAMILIES.items():
            ps = []
            for k in keys:
                if k == "C2_selected" and k not in R[src][seeds[0]]["fusion"]["results"]:
                    ps.append(1.0); continue
                d = dwf_f1 - vals(R, src, k, "macro_f1"); ps.append(S.wilcoxon_exact(d)["p"])
            for k, a_ in zip(keys, holm(ps)):
                pv[k] = a_; fam_of[k] = fam
        for label, key in T15_ROWS:
            for j, ds in enumerate(order):
                if ds in R and R[ds] and key in R[ds][sorted(R[ds])[0]]["fusion"]["results"]:
                    P("T15", label, 1 + j, ms(vals(R, ds, key, "macro_f1")))
            P("T15", label, 5, ms(vals(R, src, key, "fpr")) if key in R[src][seeds[0]]["fusion"]["results"] else "—")
            if key in R[src][seeds[0]]["fusion"]["results"]:
                P("T15", label, 6, ms(vals(R, src, key, "macro_f1") - fix_f1))
            P("T15", label, 7, "—" if key == "dwf" else (f"{pv[key]:.4f}" if key in pv else "—"))
            if key == "C2_selected":
                ks = [R[src][s]["fusion"]["results"]["C2_selected"]["selected_K"] for s in seeds]
                notes.append(f"C2 selected K per seed (validation macro-F1): {ks}")
        # decision rule (Section 3.9): DWF vs the four controls, Holm over m = 4
        ctrl = ["A1_fixed", "A2_random_regime", "A3_shuffled_entropy", "A4_var_pkt"]
        verdicts = {}
        for ds in order:
            if not R[ds]:
                continue
            d_f1 = vals(R, ds, "dwf", "macro_f1")
            rows = []
            for k in ctrl:
                d = d_f1 - vals(R, ds, k, "macro_f1"); rows.append((k, float(d.mean()), S.wilcoxon_exact(d)["p"]))
            adj = holm([r[2] for r in rows])
            gate_keys = [k for k in ("D1_stacking", "D2_gate_H", "D3_gate_H_HFR", "D4_moe") if k in R[ds][sorted(R[ds])[0]]["fusion"]["results"]]
            best_gate = max(gate_keys, key=lambda k: vals(R, ds, k, "macro_f1").mean()) if gate_keys else None
            below = None
            if best_gate:
                dd = d_f1 - vals(R, ds, best_gate, "macro_f1"); w = S.wilcoxon_exact(dd); below = bool(dd.mean() < 0 and w["p"] < 0.05)
            ok = all(r[1] > 0 for r in rows) and all(a < 0.01 for a in adj) and (below is False or below is None)
            verdicts[ds] = {"controls": [{"variant": r[0], "mean_diff_pp": r[1], "p": r[2], "p_holm_m4": a} for r, a in zip(rows, adj)],
                            "best_learned_gate": best_gate, "dwf_significantly_below_best_gate": below, "validated": ok}
        write_json(Path(cfg["out_dir"]) / "decision_rule.json", verdicts)
        allv = all(v["validated"] for v in verdicts.values())
        summ.append("## Decision rule (Section 3.9, pre-specified)\n" + "\n".join(
            f"- {ds}: {'PASS' if v['validated'] else 'FAIL'}; controls (mean Δ pp / Holm p): " +
            "; ".join(f"{c['variant']} {c['mean_diff_pp']:+.3f}/{c['p_holm_m4']:.4f}" for c in v['controls']) +
            f"; best learned gate {v['best_learned_gate']} (DWF significantly below: {v['dwf_significantly_below_best_gate']})" for ds, v in verdicts.items()))
        summ.append(f"**Overall: DWF is {'validated as an entropy-driven mechanism on all datasets' if allv else 'NOT validated on every dataset -> restate the contribution as adaptive fusion in general (per Section 3.9)'}.**"
                    " With N = 10 the smallest attainable exact Wilcoxon p is 0.00195, so a Holm family of four controls can at best reach 0.0078.")
        note15 = ("Mean ± SD (sample, n − 1) over the N master seeds; FPR and Δ are on CICIoT-2023 (Δ = variant − fixed optimal weights, paired by seed). "
                  "Holm-adj. p: exact two-sided Wilcoxon signed-rank test of DWF − variant on CICIoT-2023 macro-F1 over seeds, Holm-adjusted within each group of rows (reference, A, B, C, D, E); "
                  "'—' = not applicable. Variants of groups A–D consume identical component posteriors; competitor rows E use identical HFR vectors, partitions and seeds.")
        patches.append({"table": "T15", "note": note15})

    # ================= Table 16 (regime breakdown on the source dataset)
    if src in R and R[src]:
        seeds = sorted(R[src]); K = cfg["k_regimes"]
        rowlabels = ["Ultra-Low", "Low", "Medium", "High", "Very High"]
        for k in range(min(K, 5)):
            rr = [R[src][s]["fusion"]["regimes"][k] for s in seeds if len(R[src][s]["fusion"]["regimes"]) > k]
            lab = rowlabels[k]
            P("T16", lab, 1, ms([r["share"] * 100 for r in rr]))
            means = {}
            for ci, c in enumerate(["cnn", "bilstm", "gbdt", "dwf"]):
                v = [r[c] for r in rr if c in r]
                if v:
                    P("T16", lab, 2 + ci, ms(v)); means[c] = np.mean(v)
            if means:
                best = max(("cnn", "bilstm", "gbdt"), key=lambda c: means[c])
                P("T16", lab, 6, best.upper() if best != "bilstm" else "BiLSTM"); P("T16", lab, 7, "Yes" if best == HYP[k] else f"No (hypothesis: {HYP[k].upper() if HYP[k] != 'bilstm' else 'BiLSTM'})")
        note16 = ("CICIoT-2023 test partition, macro-F1 (%) over the classes present in each regime, mean ± SD over seeds; regimes are defined by each seed's tuned DWF boundaries "
                  "(Section 3.7); the Table 7 boundaries are those of the original single run.")
        patches.append({"table": "T16", "note": note16})

    # ================= Table 17
    for ds, nm in zip(order, ["CICIoT-2023", "Edge-IIoTset-2022", "NF-ToN-IoT-v2", "WSN-DS"]):
        if not R[ds]:
            continue
        for metric, lab, nd in (("accuracy", "Accuracy (%)", 2), ("macro_f1", "Macro-F1 (%)", 2), ("fpr", "FPR (%)", 2), ("auc_roc", "AUC-ROC", 4)):
            v = vals(R, ds, "dwf", metric)
            if not len(v):
                continue
            st = S.mean_sd_ci(v)
            key = f"{nm}|{lab}"
            P("T17", key, 3, f"{st['mean']:.{nd}f} ± {st['sd']:.{nd}f}"); P("T17", key, 4, f"[{st['ci'][0]:.{nd}f}, {st['ci'][1]:.{nd}f}]")
            P("T17", key, 5, f"{st['min']:.{nd}f} – {st['max']:.{nd}f}")
    patches.append({"table": "T17", "note": "Mean ± SD (sample SD, n − 1) over the N independent runs (master seeds, each re-drawing the split, SMOTE, initialisation and tuning); 95% CI = Student-t interval of the mean; "
                                          "AUC-ROC = macro one-vs-rest on a 100,000-instance stratified-random test subsample."})

    # ================= Table 18
    rows18 = [("P1 in-dataset tuning", order, "P1"), ("P2 frozen transfer", order[1:], "P2"), ("P3 leave-one-dataset-out", order, "P3")]
    nmap = dict(zip(order, ["CICIoT-2023", "Edge-IIoTset-2022", "NF-ToN-IoT-v2", "WSN-DS"]))
    for setting, dss, tag in rows18:
        for ds in dss:
            tr = [R[ds][s]["transfer"] for s in sorted(R[ds]) if R[ds][s]["transfer"] and tag in R[ds][s]["transfer"]]
            if not tr:
                continue
            key = f"{setting}|{nmap[ds]}"
            P("T18", key, 3, ms([t[tag]["dwf"]["accuracy"] for t in tr])); P("T18", key, 4, ms([t[tag]["dwf"]["macro_f1"] for t in tr]))
            P("T18", key, 5, ms([t[tag]["dwf"]["fpr"] for t in tr]))
            P("T18", key, 6, ms([t[tag]["dwf"]["macro_f1"] - t[tag]["fixed"]["macro_f1"] for t in tr]))
            if tag != "P1":
                P("T18", key, 7, ms([t[tag]["dwf"]["macro_f1"] - t["P1"]["dwf"]["macro_f1"] for t in tr]))
    patches.append({"table": "T18", "note": "Mean ± SD over seeds. P1: tuned on the target validation partition; P2: tuned on the CICIoT-2023 validation partition of the same seed and frozen; "
                                          "P3: tuned on the pooled validation partitions of the other three datasets (objective = mean validation macro-F1) and frozen; the target's validation and test data never enter the search. "
                                          "Δ vs fixed weights compares DWF with fixed optimal weights tuned on the same data."})

    # ================= Table 19 (adversarial)
    amap = {"FGSM (masked, projected)": "FGSM", "PGD (masked, projected)": "PGD", "DeepFool (L2-bounded, masked)": "DeepFool", "C&W (L2-bounded, masked)": "CW",
            "Black-box (HopSkipJump / Square-type)": "BlackBox", "Tree-ensemble evasion (leaf flipping / MILP)": "TreeEvasion",
            "Gate manipulation (EOT through smooth surrogate of the regime lookup)": "GateManip"}
    eps = str(cfg["adversarial"]["report_eps"])
    if src in R:
        advs = [R[src][s]["adv"] for s in sorted(R[src]) if R[src][s]["adv"]]
        for lab, an in amap.items():
            if advs and all(an in a["attacks"] and eps in a["attacks"][an] for a in advs):
                P("T19", lab, 4, ms([a["attacks"][an][eps]["unconstrained"]["ensemble_acc"] for a in advs]))
                P("T19", lab, 5, ms([a["attacks"][an][eps]["constrained"]["ensemble_acc"] for a in advs]))
        # replay (Stage 2)
        rep_rows = {**{k: v for k, v in amap.items()}, "Packet-level mutation (padding, jitter, fragmentation, interleaving)": "Mutation"}
        for lab, an in rep_rows.items():
            rj = [read_json(R[src][s]["dir"] / f"replay_{an}.json") for s in sorted(R[src]) if (R[src][s]["dir"] / f"replay_{an}.json").exists()]
            if rj:
                P("T19", lab, 3, ms([r["valid_rate"] for r in rj], 1)); P("T19", lab, 6, ms([r["acc_replayed_valid_only"] for r in rj]))
            else:
                unfilled.append(f"Table 19 '{lab}': Valid (%) and replayed accuracy need hai_dwf.replay on the original pcaps (not provided)")
        patches.append({"table": "T19", "note": f"Ensemble accuracy (%) at ε = {eps} (L∞ for FGSM/PGD/black-box/tree/gate attacks; L2 radius for DeepFool and C&W) on {advs[0]['n_eval'] if advs else 'n'} stratified-random "
                                                 f"CICIoT-2023 test windows, mean ± SD over seeds with adversarial runs. Unconstrained = all 28 components may change; constrained = controllable components only, "
                                                 f"physical bounds and dependency projection. Valid (%) and the problem-space column come from packet-level replay through the real feature extractor (Section 3.11, Stage 2)."})

    # ================= Table 20 (deployment)
    if deploy_json and Path(deploy_json).exists():
        dj = read_json(deploy_json)
        rows20 = {"Packet capture and window buffering (T = 20)": "capture", "HFR extraction, including the 28 × 10 entropy histogram": "hfr", "Scaling / quantisation": "scale",
                  "CNN inference": "cnn", "BiLSTM inference": "bilstm", "LightGBM inference": "lgbm", "DWF regime lookup and fusion": "dwf",
                  "Total cluster-head chain (extraction + inference + DWF)": "total"}
        b = dj["boards"]
        def pair(field, stage, key="mean", nd=2):
            out = []
            for bd in ("m55", "u74"):
                v = b.get(bd, {}).get(stage, {}).get(field)
                out.append(f"{v[key]:.{nd}f}" if v else None)
            return out
        for lab, st in rows20.items():
            lat = pair("latency_ms", st)
            if all(lat):
                P("T20", lab, 1, f"{lat[0]} / {lat[1]}")
            en = pair("energy_uJ", st, nd=1)
            if st == "total" and not all(en):   # total energy = sum of the per-stage energies of the same chain
                tot = []
                for bd in ("m55", "u74"):
                    parts = [b.get(bd, {}).get(x, {}).get("energy_uJ") for x in STG]
                    tot.append(f"{sum(p['mean'] for p in parts):.1f}" if all(parts) else None)
                en = tot
            if all(en):
                P("T20", lab, 4, f"{en[0]} / {en[1]}")
            for bd in ("m55", "u74"):
                pass
            mem = [b.get(bd, {}).get("_memory", {}) for bd in ("m55", "u74")]
            if all(mem):
                fl = [m.get("flash_kb", {}).get(st) for m in mem]; sr = [m.get("sram_peak_kb", {}).get(st) for m in mem]
                if all(v is not None for v in fl): P("T20", lab, 3, f"{fl[0]:.1f} / {fl[1]:.1f}")
                if all(v is not None for v in sr): P("T20", lab, 2, f"{sr[0]:.1f} / {sr[1]:.1f}")
        for lab, role in (("Radio: leaf node to cluster head, HFR payload per window", "leaf"), ("Radio: relay node to cluster head", "relay")):
            r = dj.get("radio", {}).get(role)
            if r:
                P("T20", lab, 1, f"{r['latency_ms']['mean']:.2f} (payload {r['payload_bytes']:.0f} B, {r['retx_mean']:.2f} retx)"); P("T20", lab, 4, f"{r['energy_uJ']['mean']:.1f}")
        e2e = [dj.get("e2e", {}).get(bd) for bd in ("m55", "u74")]
        if all(e2e):
            P("T20", "End-to-end detection latency (capture to decision)", 1, f"{e2e[0]['latency_ms']['mean']:.2f} / {e2e[1]['latency_ms']['mean']:.2f}")
        pw = [dj.get("power", {}).get(bd) for bd in ("m55", "u74")]
        if all(pw):
            P("T20", "Mean node power at stated decision rates", 4, " / ".join(", ".join(f"{r['mean_power_uW']:.0f} µW @ {r['rate_hz']:g} Hz" for r in p["rows"]) for p in pw))
        patches.append({"table": "T20", "note": "Measured on the target boards (M55 = Cortex-M55 + Ethos-U55, U74 = SiFive U74) with on-device cycle counters and an inline current profiler; "
                                                 "values are means over the recorded decisions; numeric format and toolchain per row are in deploy/measurements.json."})
    else:
        unfilled.append("Table 20: no deploy/measurements.json -- requires measurements on the boards (deploy/firmware/bench.c + deploy/ingest.py)")

    # ================= Appendix A2 / A3
    spec0 = {}
    for ds in order:
        p = run_dir(cfg, ds, cfg["seeds"][0]) / "hfr_spec.json"
        if p.exists():
            spec0[ds] = read_json(p)
    if spec0:
        pat = {"Flow bytes per second": r"(byt.*(s|sec)\b|bps|byte_?rate|bytes?_?per)", "LQI mean deviation": r"lqi", "Routing-table change frequency": r"rout", "Hop count": r"hop",
               "Modbus function-code distribution": r"modbus"}
        used = {ds: set() for ds in spec0}
        for lab, rx in pat.items():
            prov = set()
            for j, ds in enumerate(order):
                hit = [s for s in spec0.get(ds, []) if re.search(rx, s["name"], re.I)]
                if hit:
                    P("A2", lab, 2 + j, "✓ " + hit[0]["name"]); prov.add(hit[0]["provenance"].capitalize()); used[ds].add(hit[0]["name"])
                else:
                    P("A2", lab, 2 + j, "— (no such column)")
            P("A2", lab, 1, "/".join(sorted(prov)) if prov else "n/a")
        for j, ds in enumerate(order):
            if ds in spec0:
                P("A2", "Window entropy score H(x)", 2 + j, "✓ computed (T = 20, 28 × 10 histogram)")
                P("A2", "CNN spatial features SF-1 to SF-5", 2 + j, "✓ learned; feed the CNN head only")
        P("A2", "Window entropy score H(x)", 1, "Engineered"); P("A2", "CNN spatial features SF-1 to SF-5", 1, "Learned")
        for j, ds in enumerate(order):
            rest = [s["name"] for s in spec0.get(ds, []) if s["name"] not in used.get(ds, set())]
            P("A2", "Remaining components (to reach 28)", 2 + j, f"{len(rest)}: " + ", ".join(rest[:8]) + (" …" if len(rest) > 8 else ""))
        P("A2", "Remaining components (to reach 28)", 1, "Raw / engineered")
        stable = []
        for ds in order:
            sp = [read_json(run_dir(cfg, ds, s) / "hfr_spec.json") for s in cfg["seeds"] if (run_dir(cfg, ds, s) / "hfr_spec.json").exists()]
            stable.append(all([x["name"] for x in q] == [x["name"] for x in sp[0]] for q in sp) if sp else None)
        summ.append(f"HFR component selection identical across seeds (per dataset, order {order}): {stable}  (Table A2 lists seed {cfg['seeds'][0]}; full specs in runs/*/seed*/hfr_spec.json)")
        patches.append({"table": "A2", "note": "Generated from runs/<dataset>/seed0/hfr_spec.json: components ranked by mutual information with the label on the fitting partition only; engineered components fill up when fewer than 28 raw columns exist."})

    sd0 = all_stage_seeds(cfg["seeds"][0]); sha = {}
    for ds in order:
        p = run_dir(cfg, ds, cfg["seeds"][0]) / "partitions.sha256"
        if p.exists():
            sha[ds] = p.read_text().strip()
    man = R[src][sorted(R[src])[0]]["manifest"] if src in R and R[src] else None
    P("A3", "Stage seeds", 1, "Derived as SHA-256(\"s0:stage\") mod 2^31−1; seed " + str(cfg["seeds"][0]) + ": " + ", ".join(f"{k}={v}" for k, v in sd0.items()) + "; all seeds in runs/*/seed*/manifest.json")
    if sha:
        P("A3", "Partition files", 1, "Index files runs/<dataset>/seed<k>/partitions.npz; seed " + str(cfg["seeds"][0]) + " SHA-256 prefixes: " + ", ".join(f"{k}={v[:12]}" for k, v in sha.items()))
        P("A3", "Partition files", 2, "Recorded (partitions.sha256)")
    if man:
        v = man["versions"]
        P("A3", "Software versions", 1, ", ".join(f"{k} {v[k]}" for k in ("python", "tensorflow", "lightgbm", "optuna", "xgboost", "catboost", "imblearn") if k in v) + "; adversarial attacks: in-house (hai_dwf/adversarial.py)")
        P("A3", "Software versions", 2, "Recorded in manifest.json")
    if repo_url:
        P("A3", "Code, weights and splits", 1, repo_url); P("A3", "Code, weights and splits", 2, "Released")
    else:
        unfilled.append("Table A3 'Code, weights and splits': pass --repo-url once the repository / DOI exists")
    P("A3", "Stage seeds", 2, "Recorded")

    # ================= summary
    lines = ["# Results summary (auto-generated)", ""] + summ + [""] + (["## Notes"] + [f"- {n}" for n in notes] if notes else [])
    lines += ["", "## Cells NOT filled (no data behind them)"] + ([f"- {u}" for u in unfilled] if unfilled else ["- none"])
    return patches, unfilled, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg"); ap.add_argument("--deploy"); ap.add_argument("--repo-url"); ap.add_argument("--out", default="results")
    a = ap.parse_args(); cfg = load_cfg(a.cfg); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    patches, unfilled, md = build(cfg, a.deploy, a.repo_url)
    write_json(out / "patches.json", patches); (out / "results_summary.md").write_text(md)
    print(md); print(f"\n{len(patches)} patches -> {out/'patches.json'}")


if __name__ == "__main__":
    main()
