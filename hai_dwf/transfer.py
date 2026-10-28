"""Cross-dataset transfer protocol of Section 3.10 (P1-P4), leakage-free by construction.

P1: parameters tuned on the validation partition of the target.
P2: parameters tuned on the CICIoT-2023 validation partition of the same master seed, applied unchanged.
P3: parameters tuned on the POOLED validation partitions of the other three datasets (objective = mean macro-F1 over
    the three validation sets), applied unchanged; nothing from the target (validation or test) is used.
P4: fixed-weight (K = 1) and equal-weight baselines evaluated under the same conditions (tuned under the same data as P1/P2/P3).
Test partitions are touched only after the parameters are frozen.

    python -m hai_dwf.transfer --cfg config/default.yaml --seed 0
"""
from __future__ import annotations
import argparse
import numpy as np

from .common import load_cfg, stage_seed, run_dir, write_json, read_json, evaluate
from . import fusion as Fu


def _set(cfg, ds, seed, split="val"):
    rd = run_dir(cfg, ds, seed); Z = Fu.load_comp(rd)
    return Z, {"P": Fu.stack3(Z, "val"), "y": Z["y_val"], "sig": Z["sig_shannon10_val"], "C": Z["p_cnn_val"].shape[1]}


def run(cfg: dict, seed: int, force=False):
    order = cfg["dataset_order"]; src = order[0]
    Zs, sets = {}, {}
    for ds in order:
        Zs[ds], sets[ds] = _set(cfg, ds, seed)
    K = cfg["k_regimes"]; bud = cfg["budgets"]["dwf"]; cap = cfg["tune_rows"]["fusion"]
    pooled_params = {}
    for tgt in order:
        rd = run_dir(cfg, tgt, seed)
        if (rd / "transfer.json").exists() and not force:
            continue
        Z = Zs[tgt]; C = sets[tgt]["C"]; benign = int(Z["benign"]) if int(Z["benign"]) >= 0 else None
        Pt, st, y = Fu.stack3(Z, "test"), Z["sig_shannon10_test"], Z["y_test"]

        def ev(par):
            return evaluate(y, Fu.apply_dwf(par, Pt, st).argmax(1), C, benign, cfg["fpr_mode"])
        out = {"target": tgt, "seed": seed}
        # P1
        p1 = read_json(rd / "dwf_params.json")
        out["P1"] = {"dwf": ev(p1["dwf"]), "fixed": ev(p1["A1_fixed"]), "params": p1["dwf"]}
        out["P1"]["equal"] = evaluate(y, Pt.mean(1).argmax(1), C, benign, cfg["fpr_mode"])
        # P2 (frozen CICIoT-2023 parameters)
        if tgt != src:
            ps = read_json(run_dir(cfg, src, seed) / "dwf_params.json")
            out["P2"] = {"dwf": ev(ps["dwf"]), "fixed": ev(ps["A1_fixed"]), "equal": out["P1"]["equal"], "params": ps["dwf"]}
        # P3 (leave-one-dataset-out, pooled validation of the other datasets)
        others = [d for d in order if d != tgt]
        pool = [sets[d] for d in others]
        pd_ = tune_pair(pool, K, seed, bud, cap)
        out["P3"] = {"dwf": ev(pd_["dwf"]), "fixed": ev(pd_["fixed"]), "equal": out["P1"]["equal"], "params": pd_["dwf"], "tuned_on": others}
        # entropy-scale comparability (Section 3.10)
        out["entropy_quantiles"] = {d: np.percentile(Zs[d]["sig_shannon10_test"], [5, 25, 50, 75, 95]).tolist() for d in order}
        write_json(rd / "transfer.json", out)
        print(f"[done] transfer {tgt} seed {seed}: P1={out['P1']['dwf']['macro_f1']:.3f} P3={out['P3']['dwf']['macro_f1']:.3f}")


def tune_pair(sets, K, seed, trials, cap):
    return {"dwf": Fu.tune_dwf(sets, K, seed, trials, True, None, cap, "P3_dwf"),
            "fixed": Fu.tune_dwf(sets, 1, seed, trials, True, None, cap, "P3_fixed")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", default=None)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); run(load_cfg(a.cfg), a.seed, a.force)


if __name__ == "__main__":
    main()
