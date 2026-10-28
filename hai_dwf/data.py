"""Dataset loading, block-wise leakage-free partitioning, HFR construction, preprocessing, windowing.

Partitioning (Section 3.4 / Table A1): rows are grouped into *blocks* (one block = one capture session if a session column
exists, otherwise `block_size` consecutive rows).  Blocks are split, stratified by their majority label, into
test (20 %) and training allocation (80 %); the training allocation is split again into fitting (90 % -> 72 % overall) and
validation (10 % -> 8 % overall).  Windows of length T are formed *inside* a block, so no window straddles two
partitions and temporal neighbours never leak across the split.  All statistics (MICE, robust scaler, min-max for the
entropy histogram, SMOTE, regime boundaries) are fitted on the fitting partition only.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .common import stage_seed, sha256_file, write_json


# ----------------------------------------------------------------------------- loading
def load_raw(cfg: dict, ds: str, max_rows: int | None = None) -> pd.DataFrame:
    d = cfg["datasets"][ds]
    p = Path(d["path"])
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if d.get("order_col"):
        df = df.sort_values(d["order_col"], kind="stable").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    return df


# ----------------------------------------------------------------------------- partitions
def block_ids(df: pd.DataFrame, dcfg: dict, block_size: int) -> np.ndarray:
    if dcfg.get("session_col") and dcfg["session_col"] in df:
        # consecutive runs of the same session id form one block (keeps temporal order inside the block)
        s = df[dcfg["session_col"]].to_numpy()
        change = np.r_[True, s[1:] != s[:-1]]
        return np.cumsum(change) - 1
    return np.arange(len(df)) // block_size


def make_partitions(df: pd.DataFrame, dcfg: dict, cfg: dict, seed: int, classes: list) -> dict:
    y = df[dcfg["label_col"]].map({c: i for i, c in enumerate(classes)}).to_numpy()
    b = block_ids(df, dcfg, cfg["block_size"])
    nb = b.max() + 1
    # majority label per block
    maj = np.zeros(nb, dtype=np.int64)
    for blk, grp in pd.Series(y).groupby(b):
        maj[blk] = np.bincount(grp.to_numpy()).argmax()
    # merge classes with <3 blocks into a pseudo-class so that stratification is defined
    cnt = np.bincount(maj, minlength=len(classes))
    strat = np.where(cnt[maj] < 3, -1, maj)
    rs = stage_seed(seed, "split")
    idx_blocks = np.arange(nb)
    sss1 = StratifiedShuffleSplit(1, test_size=cfg["splits"]["test"], random_state=rs)
    try:
        tr_b, te_b = next(sss1.split(idx_blocks, strat))
    except ValueError:  # too few blocks per stratum
        rng = np.random.default_rng(rs); perm = rng.permutation(nb); k = int(nb * cfg["splits"]["test"])
        te_b, tr_b = perm[:k], perm[k:]
    sss2 = StratifiedShuffleSplit(1, test_size=cfg["splits"]["val_of_train"], random_state=rs + 1)
    try:
        fi, vi = next(sss2.split(tr_b, strat[tr_b]))
    except ValueError:
        rng = np.random.default_rng(rs + 1); perm = rng.permutation(len(tr_b)); k = int(len(tr_b) * cfg["splits"]["val_of_train"])
        vi, fi = perm[:k], perm[k:]
    which = np.full(nb, -1)
    which[tr_b[fi]] = 0; which[tr_b[vi]] = 1; which[te_b] = 2
    part = which[b]                       # 0 fit / 1 val / 2 test per row
    return {"part": part.astype(np.int8), "block": b.astype(np.int64), "y": y.astype(np.int64)}


def save_partitions(P: dict, rd: Path):
    np.savez_compressed(rd / "partitions.npz", **P)
    sha = sha256_file(rd / "partitions.npz")
    (rd / "partitions.sha256").write_text(sha + "\n")
    counts = {n: int((P["part"] == i).sum()) for i, n in enumerate(["fit", "val", "test"])}
    write_json(rd / "partition_counts.json", {"counts": counts, "total": int(len(P["part"])), "sha256": sha})
    return sha


# ----------------------------------------------------------------------------- HFR construction
def fit_hfr_spec(df_fit: pd.DataFrame, dcfg: dict, n_out: int, seed: int, override: str | None = None, mode: str = "default") -> list[dict]:
    """Return the ordered list of the n_out HFR components for this dataset (feeds Table A2).

    1. If `override` (yaml) is given: use it verbatim.
    2. Otherwise: numeric non-constant columns are ranked by mutual information with the label on the FITTING partition
       only; the top n_out are kept ("raw").  If fewer than n_out numeric columns exist, the remainder are filled with
       deterministic engineered components (log1p of the top columns, then pairwise ratios of the top columns).
    """
    if override:
        import yaml
        with open(override) as f:
            return yaml.safe_load(f)["components"]
    from sklearn.feature_selection import mutual_info_classif
    lab = dcfg["label_col"]
    num = [c for c in df_fit.columns if c != lab and c != dcfg.get("session_col") and c != dcfg.get("order_col")
           and pd.api.types.is_numeric_dtype(df_fit[c])]
    num = [c for c in num if df_fit[c].nunique() > 1]
    if mode == "no_wsn":      # control: drop the WSN-specific measurements (LQI, hop count, routing changes)
        num = [c for c in num if not re.search(r"(lqi|hop|rout)", c, re.I)]
    sub = df_fit.sample(min(len(df_fit), 200_000), random_state=stage_seed(seed, "split"))
    Xs = sub[num].fillna(sub[num].median()).to_numpy(dtype=np.float64)
    mi = mutual_info_classif(Xs, sub[lab].astype("category").cat.codes.to_numpy(), random_state=0)
    rank = [num[i] for i in np.argsort(-mi)]
    spec = [{"name": c, "provenance": "raw", "source": [c], "op": "identity"} for c in rank[:n_out]]
    if mode == "raw_only":    # control: no engineered components; unused slots are constant zeros so the 4x7 layout is unchanged
        spec += [{"name": f"pad{j}", "provenance": "pad", "source": [spec[0]["source"][0]], "op": "zero"} for j in range(n_out - len(spec))]
        return spec[:n_out]
    top = [s["source"][0] for s in spec]
    i = 0
    while len(spec) < n_out and i < len(top):
        spec.append({"name": f"log1p({top[i]})", "provenance": "engineered", "source": [top[i]], "op": "log1p_signed"}); i += 1
    i = 0
    while len(spec) < n_out and len(top) > 1:
        a, b = top[i % len(top)], top[(i + 1) % len(top)]
        spec.append({"name": f"{a}/{b}", "provenance": "engineered", "source": [a, b], "op": "ratio"}); i += 1
        if i > 4 * len(top):
            break
    return spec[:n_out]


def apply_hfr(df: pd.DataFrame, spec: list[dict]) -> np.ndarray:
    out = np.empty((len(df), len(spec)), dtype=np.float32)
    for j, s in enumerate(spec):
        x = df[s["source"][0]].to_numpy(dtype=np.float64, na_value=np.nan)
        if s["op"] == "zero":
            out[:, j] = 0.0; continue
        if s["op"] == "log1p_signed":
            x = np.sign(x) * np.log1p(np.abs(x))
        elif s["op"] == "ratio":
            z = df[s["source"][1]].to_numpy(dtype=np.float64, na_value=np.nan)
            x = x / np.where(np.abs(z) < 1e-9, np.nan, z)
        out[:, j] = x
    return out


# ----------------------------------------------------------------------------- preprocessing
class Preproc:
    """MICE -> class-agnostic robust scaling clipped to [-1, 1]; plus min-max for the entropy histogram."""

    def __init__(self, iters: int, seed: int):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer, SimpleImputer
        self.has_nan = None
        self.imp = IterativeImputer(max_iter=iters, random_state=stage_seed(seed, "mice"), sample_posterior=False,
                                    initial_strategy="median", skip_complete=True)
        self.fallback = SimpleImputer(strategy="median")

    def fit(self, X):
        self.has_nan = bool(np.isnan(X).any())
        Xf = X
        if self.has_nan:
            sub = X[np.random.default_rng(0).permutation(len(X))[:100_000]]
            self.imp.fit(sub)
            Xf = self.transform_imp(X)
        self.med = np.median(Xf, 0)
        self.raw_min, self.raw_max = Xf.min(0), Xf.max(0)
        q75, q25 = np.percentile(Xf, [75, 25], axis=0)
        self.iqr = np.where((q75 - q25) < 1e-9, 1.0, q75 - q25)
        Z = self._scale(Xf)
        self.mn, self.mx = Z.min(0), Z.max(0)          # fitting-partition min/max for the entropy histogram
        self.mx = np.where(self.mx - self.mn < 1e-9, self.mn + 1.0, self.mx)
        return self

    def transform_imp(self, X):
        if not self.has_nan or not np.isnan(X).any():
            return X
        return self.imp.transform(X).astype(np.float32)

    def _scale(self, X):
        return np.clip((X - self.med) / self.iqr, -1.0, 1.0).astype(np.float32)

    def transform(self, X):
        return self._scale(self.transform_imp(X))

    def unit(self, Z):  # scaled [-1,1] -> min-max [0,1] using FITTING min/max
        return np.clip((Z - self.mn) / (self.mx - self.mn), 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- windows
def window_index(block: np.ndarray, T: int) -> np.ndarray:
    """(N, T) row indices into the partition array; windows never cross block boundaries (edge-padded at block start)."""
    N = len(block)
    start = np.r_[0, np.flatnonzero(block[1:] != block[:-1]) + 1]
    bs = np.repeat(start, np.diff(np.r_[start, N]))
    pos = np.arange(N) - bs
    off = np.arange(-(T - 1), 1)
    return bs[:, None] + np.maximum(pos[:, None] + off[None, :], 0)


def gather(X: np.ndarray, idx: np.ndarray, sl: slice | np.ndarray) -> np.ndarray:
    return X[idx[sl]]


def iter_chunks(n: int, size: int):
    for a in range(0, n, size):
        yield slice(a, min(a + size, n))
