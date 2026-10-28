# HAI+DWF validation pipeline (answers reviewer comments 1–5)

Everything needed to **produce the numbers for the TBR/TBC cells** of the revised manuscript (Tables 15–20, A2, A3) from the four
datasets and the target boards, and to write them into the `.docx` with the yellow highlight kept.
No number in the manuscript is estimated or simulated by this code; if a run or a measurement does not exist, the cell stays
TBR/TBC and `results/results_summary.md` says why.

## What you must supply (cannot be generated)
| Need | Used for |
|---|---|
| The four datasets as one table each (`parquet` or `csv`), with a `label` column; optional `session_col` / `order_col` | Tables 15–18 |
| The original packet captures (pcaps) + the dataset's feature extractor (e.g. CICFlowMeter) | Table 19 `Valid (%)` and problem-space columns (`hai_dwf/replay.py`) |
| Measurements from the boards (cycle counters, current profiler, linker map, radio logs) | Table 20 (`deploy/`) |
| Repository URL / DOI | Table A3 last row (`--repo-url`) |

## Quick start
```bash
pip install -r requirements.txt
python tests/test_units.py                                 # statistics / entropy / windowing unit tests
# 1. edit config/default.yaml: dataset paths, label columns, benign labels (nothing else needs changing)
python run_all.py --cfg config/default.yaml                # components -> fusion -> transfer -> adversarial -> aggregate
# 2. board measurements -> deploy/measurements.json (see deploy/ingest.py header for the CSV formats)
python deploy/ingest.py --dir <measurement dir> --boards m55 u74 --vdd 3.3 --idle-uA <value>
# 3. aggregate (re-run after step 2) and fill the manuscript
python -m hai_dwf.aggregate --cfg config/default.yaml --deploy deploy/measurements.json --repo-url <URL> --out results
python -m hai_dwf.fill_docx --docx IASC_-Adnan_revised.docx --patches results/patches.json --out IASC_-Adnan_final.docx
```
Smoke test of the code path on synthetic data (NOT for the paper): `python tests/make_synthetic.py data && python run_all.py --cfg config/smoke.yaml`.

## Pipeline (one process per dataset × master seed; resumable)
| Stage | Module | Output (per `runs/<dataset>/seed<k>/`) | Manuscript |
|---|---|---|---|
| 1 | `components.py` | `components.npz` (posteriors of CNN, BiLSTM, LightGBM, E1–E3, MoE; control signals), models, `partitions.npz` + sha256, `hfr_spec.json`, `manifest.json` | §3.1, 3.4, 3.5–3.6, Tables A1–A3 |
| 2 | `fusion.py` | `fusion.json`, `fusion_preds.npz`, `dwf_params.json` — every variant A1–E3 on **identical** posteriors | §3.9, Tables 15, 16, 17 |
| 3 | `transfer.py` | `transfer.json` — P1, P2, P3 (+ fixed/equal baselines = P4) | §3.10, Table 18 |
| 4 | `adversarial.py`, `replay.py` | `adversarial.json`, `replay_<attack>.json` | §3.11, Table 19 |
| 5 | `deploy/*`, `ingest.py` | `deploy/measurements.json` | §3.12, Table 20 |
| 6 | `aggregate.py`, `fill_docx.py` | `results/patches.json`, `results/results_summary.md`, final `.docx` | all |

Every stochastic stage has its own seed derived from the master seed (`common.stage_seed`): split, MICE, SMOTE, initialisation,
shuffling, LightGBM bagging, TPE, adversarial generation, bootstrap. Splits are **block-wise** (a block = one capture session, or
`block_size` consecutive rows) and windows are built inside a block, so no window or session straddles two partitions; all
statistics (MICE, robust scaler, histogram min/max, SMOTE, regime boundaries) are fitted on the fitting partition / validation
partition only; the test partition is touched once, after all choices are frozen.

## Decisions I had to make (please confirm against the manuscript)
1. **Entropy definition / Table 7 boundaries.** With the pooled 28×10 histogram of §3.7, H(x) lies in **[ln 28, ln 280] = [3.33, 5.63] nats** (every component
   constant ⇒ 3.33). The boundaries 0.6 / 1.4 / 2.2 / 3.0 in Table 7 therefore cannot come from this definition. Either the original entropy was defined
   differently (e.g. per-component, or on a single packet-size histogram) or the boundaries are not in nats. The pipeline learns boundaries inside the
   observed range, so the experiments are valid, but **§3.7 and Table 7 must be reconciled** (the document currently states the pooled definition).
2. **Smallest attainable Wilcoxon p** with N = 10 is 2/2¹⁰ = 0.00195. Holm over a family of *m* tests gives at best m·0.00195, so the pre-specified rule
   (Holm p < 0.01) is reachable only for m ≤ 5. `aggregate.py` therefore evaluates the decision rule with the four controls (m = 4, best 0.0078) and Holm
   within each group (A–E) for Table 15. Use N ≥ 15 seeds if you want a larger family.
3. **FPR** is the macro-averaged one-vs-rest false-positive rate (`fpr_mode: macro_ovr`). If the paper meant "benign flagged as attack", set `benign_vs_attack`.
4. **CNN head.** SF-1…SF-5 feed only a softmax head; the GBDT and the DWF see the 28 HFR vector (Table A2 says so).
5. **MoE (D4)** replaces the non-differentiable GBDT expert by an MLP expert (trees cannot be trained jointly by gradient).
6. **Learned gate / stacking budgets** are `budgets.gate`, `budgets.stack` (500 in `default.yaml`, as for DWF). `aggregate.py` warns if budgets differ across variants.
7. **Table 15 p-value column** = exact two-sided Wilcoxon of (DWF − variant) on CICIoT-2023 macro-F1 over seeds, Holm within the group; Δ = variant − fixed weights.
8. **SMOTE cap**: `smote.cap_per_class` (20,000) keeps the balanced windows in RAM (≈0.7 GB for 16 classes). Raise it if you have the memory; majority classes are under-sampled to the cap.

## Cost (order of magnitude, 2 × RTX A6000, default config)
Stage 1 dominates (BiLSTM, 300-trial LightGBM, E1–E3, MoE): several hours per (dataset, seed) on CICIoT-2023; hours for the smaller sets.
Use `--datasets/--seeds` to parallelise over machines; the stages are independent per seed except `transfer` (needs the four datasets of a seed).

## Not covered / limits (stated, not hidden)
* HopSkipJump and MILP evasion are replaced by a Square-type random-search attack and a greedy threshold-crossing attack (documented in `adversarial.py`).
* `replay.py` needs the pcaps and an extractor that reproduces the dataset's features; without them Table 19 `Valid (%)` stays TBR.
* Board numbers (Table 20) can only come from the boards; `deploy/firmware/bench.c` is a measurement skeleton with three hooks to fill in.

## Tables 21-23 (added)
`python -m hai_dwf.extra --cfg config/default.yaml --out results_ext` after producing the runs it reads:
Table 21 = `components`+`fusion` once per `resampling: none|ros|smote|borderline` (separate `out_dir` each, see `config/t21_*.yaml`, `run_t21.sh`);
Table 23 raw-only / no-WSN controls = runs with `hfr_mode: raw_only|no_wsn` (`config/t23_*.yaml`, `run_ext.sh`); Tier 1 (int8 GBDT) and the LEACH rotation are computed inside `extra.py`;
Table 22 = IG / TreeSHAP / gate permutation on the stored models. Merge `results_ext/patches_ext.json` with `results/patches.json` and run `fill_docx`.
