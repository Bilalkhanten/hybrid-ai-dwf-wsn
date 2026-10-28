# HAI+DWF: entropy-adaptive hybrid ensemble for intrusion detection in wireless sensor networks

A CNN, a BiLSTM with temporal attention and a LightGBM model each classify network traffic features (the CNN and LightGBM use the
latest time step, the BiLSTM a window of 20 steps). A **dynamic weighted fusion (DWF)** layer then combines their class
posteriors, choosing the weights per window from the Shannon entropy of that window. This repository is the implementation: data handling, the models, the fusion layer and its ablations, transfer
and adversarial evaluation, explainability, statistics, and export for resource-constrained boards.

```
 raw table ─▶ 28 features (HFR) ─▶ windows of T = 20 steps
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
        CNN (last step)      BiLSTM + attention (window)   LightGBM (last step)
              │ p₁                      │ p₂                      │ p₃
              └─────────────┬───────────┴───────────┬─────────────┘
                            │                       │
        window entropy H ─▶ regime r(H) ─▶ weights W[r] ─▶  p = Σₖ W[r,k] · pₖ  ─▶ class
```

## How it works
1. **Features (HFR).** The 28 components per time step are the numeric columns with the highest mutual information with the
   label (computed on the fitting partition only). If a dataset has fewer than 28 usable columns, the rest are filled with
   deterministic engineered components (signed `log1p`, then pairwise ratios). A YAML override can fix the list per dataset
   (`hfr_overrides`).
2. **Component models** (`hai_dwf/models.py`, `components.py`)
   - **CNN:** the 28-vector is reshaped to 4×7×1 → Conv 32 (3×3) → BatchNorm → MaxPool → Conv 64 (1×1) → BatchNorm → global
     average pooling → Dense 32 → Dropout 0.25 → a 5-unit layer → softmax over the classes.
   - **BiLSTM:** three stacked bidirectional LSTMs (128 / 64 / 48 units, dropout 0.30 / 0.25 / 0.20) → scaled dot-product
     temporal self-attention (mean-pooled) → Dense 64 → softmax.
   - **LightGBM:** hyper-parameters tuned with Optuna TPE on the validation partition, then refitted on all fitting rows.
3. **Window entropy** (`entropy.py`). Each window of unit-scaled features is turned into a pooled 28 × B histogram (B = 10 bins by
   default); its Shannon entropy H is the control signal. Rényi-2, Tsallis-2, other bin counts, packet-size variance and the
   ensemble's predictive entropy are implemented as alternatives.
4. **Dynamic weighted fusion** (`fusion.py`). The range of H is cut into K regimes (default 5). Regime boundaries and one normalised
   weight vector per regime (three weights summing to 1) are searched with Optuna TPE to maximise validation macro-F1; the test partition is touched once, after
   everything is frozen.

### Fusion variants implemented on identical posteriors
Stage 1 stores every component's posteriors once, and Stage 2 evaluates all variants on exactly those, so differences come from
the fusion rule only.

| Family | Variants |
|---|---|
| Reference | equal-weight average, standalone CNN / BiLSTM / LightGBM |
| A. Controls | fixed weights (K = 1), random regime, shuffled entropy, packet-size variance, constant signal |
| B. Other signals | Rényi-2, Tsallis-2, Shannon with 5 / 20 / 50 bins, predictive entropy |
| C. Regime structure | quantile boundaries, K selected on validation |
| D. Learned fusion | stacking (logistic regression on log posteriors), learned gate on entropy, learned gate on entropy + features, jointly trained mixture of experts, Dempster–Shafer, confidence-weighted |
| E. Competitors | CNN + multi-head attention, BiLSTM embedding + random forest, tuned XGBoost / CatBoost |

## Evaluation protocols
- **Leakage-free splits.** Rows are grouped into blocks (one capture session, or `block_size` consecutive rows). Blocks are split
  into test (20 %) and fit / validation (72 % / 8 % overall), stratified by majority label, and windows are built inside a block, so
  no window or session straddles two partitions. MICE imputation, robust scaling, histogram ranges, resampling and regime
  boundaries are all fitted on the fitting partition only. Partitions are stored with a SHA-256.
- **Class imbalance.** The fitting partition is class-capped and balanced with SMOTE (`resampling: smote`); `none` (class weights),
  `ros` (random oversampling) and `borderline` are available for comparison.
- **Cross-dataset transfer** (`transfer.py`). P1: tune on the target's validation set. P2: tune on the source dataset and apply
  unchanged. P3: tune on the pooled validation sets of the other datasets and apply unchanged. P4: fixed-weight and equal-weight
  baselines under the same conditions.
- **Adversarial robustness** (`adversarial.py`, `replay.py`). Perturbations act on the scaled 28 × T windows, either unconstrained or
  constrained to attacker-controllable components (regex in the config) with physical bounds and re-derived engineered components.
  White-box FGSM, PGD, DeepFool and Carlini-Wagner L2 use CNN + BiLSTM as a differentiable substitute and are transferred to the
  full ensemble; also implemented are a decision-only Square-type attack, a tree-evasion attack on LightGBM and an EOT
  gate-manipulation attack through a smooth entropy surrogate. `replay.py` realises a feature-space target at packet level and
  re-extracts the features, to measure how many adversarial vectors are actually feasible.
- **Explainability and controls** (`extra.py`). Integrated Gradients (CNN, BiLSTM), TreeSHAP (LightGBM), entropy-permutation importance of the gate, rank
  stability across seeds, resampling sensitivity, raw-only and no-WSN feature controls, an int8 LightGBM and a simulated
  cluster-head rotation on the test traffic.
- **Statistics** (`stats.py`). Paired DeLong AUC test, exact McNemar, cluster bootstrap for accuracy differences, exact Wilcoxon over seeds
  with Holm correction, Cohen's d_z.

## Edge deployment (`deploy/`)
`export_models.py` converts the CNN and BiLSTM to TensorFlow Lite int8 and LightGBM to dependency-free C, and writes `hfr_extract.h`
(the histogram entropy and scaling constants) so feature extraction runs on the device with the host's numbers.
`firmware/bench.c` is a Zephyr benchmark skeleton (Cortex-M55 + Ethos-U55 and RISC-V U74) with three hooks to fill in. It times each
stage with the cycle counter and marks GPIO for a current profiler. `ingest.py` turns the recorded CSV files into
`deploy/measurements.json`. Nothing in `deploy/` estimates a number: latency, energy and memory come from the board.

## Quick start
```bash
pip install -r requirements.txt
python tests/test_units.py                        # statistics, entropy and windowing unit tests (seconds)

# end-to-end code-path check on SYNTHETIC data (about 20 minutes on a CPU; the numbers mean nothing)
python tests/make_synthetic.py data
python run_all.py --cfg config/smoke.yaml
```

### Run on your own data
1. Provide one table per dataset (`parquet` or `csv`) with a label column, and edit `config/default.yaml`: paths, `label_col`,
   `benign_label`, and optionally `session_col` / `order_col`.
2. Run the stages (each (dataset, seed) is its own process and finished jobs are skipped, so runs are resumable):
```bash
python run_all.py --cfg config/default.yaml                          # components -> fusion -> transfer -> adversarial -> aggregate
python run_all.py --cfg config/default.yaml --datasets wsn_ds --seeds 0 1 2    # a subset, e.g. to spread jobs over machines
```
or one stage at a time:
```bash
python -m hai_dwf.components --cfg config/default.yaml --dataset ciciot2023 --seed 0   # train components, store posteriors
python -m hai_dwf.fusion     --cfg config/default.yaml --dataset ciciot2023 --seed 0   # every fusion variant
python -m hai_dwf.transfer   --cfg config/default.yaml --seed 0                        # needs all datasets of that seed
python -m hai_dwf.adversarial --cfg config/default.yaml --dataset ciciot2023 --seed 0
python -m hai_dwf.extra      --cfg config/default.yaml --out results_ext               # attribution and control experiments
```
`extra.py` reads the runs of the resampling and feature-control comparisons, which `run_t21.sh` (`config/t21_*.yaml`) and
`run_ext.sh` (`config/t23_*.yaml`) produce.
Full runs are expensive: with the default configuration (300 + 500 TPE trials, a BiLSTM, three components plus competitors) expect
hours per (dataset, seed) on a GPU for the large datasets. Lower `budgets`, `tune_rows` and `epochs` for a quick trial; `config/smoke.yaml` is the
smallest profile.

## Outputs
```
runs/<dataset>/seed<k>/
  components.npz        posteriors of every model, control signals, labels            (Stage 1)
  cnn.keras bilstm.keras lgbm.txt preproc.pkl hfr_spec.json partitions.npz           models, preprocessing, feature spec
  manifest.json         seeds, partition SHA-256, timings, per-component metrics, library versions
  fusion.json  fusion_preds.npz  dwf_params.json                                      (Stage 2)
  transfer.json                                                                       (Stage 3)
  adversarial.json  replay_<attack>.json                                              (Stage 4)
results/                aggregated tables, statistics and a plain-language summary     (aggregate.py)
```

## Configuration
| Key | Default | Meaning |
|---|---|---|
| `window` / `bins` / `n_hfr` | 20 / 10 / 28 | window length T, histogram bins per component, number of features |
| `k_regimes` | 5 | entropy regimes in the DWF |
| `budgets` | 300 / 500 / 500 / 500 / 300 | Optuna trials: LightGBM, DWF, learned gate, stacking, XGBoost/CatBoost |
| `splits`, `block_size` | 20 % test, 10 % of the rest validation; 200 rows | partitioning |
| `resampling`, `smote` | `smote`; cap 20 000 per class, k = 5 | imbalance handling on the fitting partition |
| `hfr_mode` | `default` | `raw_only` or `no_wsn` for feature-ablation controls |
| `fpr_mode` | `macro_ovr` | macro one-vs-rest false-positive rate, or `benign_vs_attack` |
| `adversarial` | ε ∈ {0.02, 0.05, 0.10}, 20 PGD steps | attack strengths, controllable / frozen component regexes |
| `seeds` | 0 … 9 | master seeds; every stage derives its own seed from the master seed (`common.stage_seed`) |

## Repository layout
| Path | Contents |
|---|---|
| `hai_dwf/data.py` | loading, block-wise partitions, feature construction, imputation and scaling, windowing |
| `hai_dwf/models.py`, `components.py` | network definitions, training, posterior storage (Stage 1) |
| `hai_dwf/entropy.py`, `fusion.py` | control signals, DWF, all fusion variants (Stage 2) |
| `hai_dwf/transfer.py` | cross-dataset transfer protocols P1–P4 |
| `hai_dwf/adversarial.py`, `replay.py` | adversarial attacks, problem-space replay |
| `hai_dwf/extra.py` | explainability, resampling and feature-control experiments |
| `hai_dwf/stats.py`, `common.py` | statistical tests, metrics, seeding, run directories |
| `hai_dwf/aggregate.py` | collects all runs into tables and a results summary |
| `deploy/` | model export, board-measurement ingestion, firmware benchmark skeleton |
| `config/` | `default.yaml`, the `smoke*.yaml` code-path profiles, resampling and feature-control profiles |
| `tests/` | unit tests and the synthetic-data generator |

## Design notes
- **Entropy range.** The 28 per-component histograms are pooled into one distribution, so H lies in [ln 28, ln 280] = [3.33, 5.63] nats for B = 10
  (3.33 when every component is constant). Regime boundaries are therefore learned inside the observed range instead of being fixed in advance.
- **Statistical power with few seeds.** With N = 10 seeds the smallest attainable exact two-sided Wilcoxon p is 2/2¹⁰ ≈ 0.00195, and Holm over a
  family of m tests can do no better than m · 0.00195, so a strict threshold is only reachable for small families; use more seeds for larger ones.
- **CNN head.** The 5-unit layer feeds only a softmax head; LightGBM and the DWF see the 28-vector.
- **Mixture of experts.** Trees cannot be trained by gradient, so the third expert of the jointly trained mixture is an MLP.
- **Cost of tuning.** The learned gate and stacking use `budgets.gate` and `budgets.stack` (500 by default, like the DWF); the aggregation warns if budgets differ across variants.
- **Adversarial coverage.** The decision-only attack is a Square-type random search and the tree attack is a greedy threshold-crossing search, not
  HopSkipJump or MILP; they are documented in `adversarial.py`.
- **Replay needs the original captures** and the exact feature extractor of the dataset (`--extractor module:function`); without them only
  feature-space results are reported.

## Reporting utilities (optional)
`hai_dwf.aggregate` writes `results/patches.json` and `results/results_summary.md`; `hai_dwf.fill_docx` can write those numbers into a Word document, and
`make_figures.py` plots the summary tables from a filled `results_template.json` (it refuses to run while any value is missing). `diagrams.py` draws the
protocol schematics. None of them estimate a number: a value without a run behind it is left unfilled and listed in the summary.
