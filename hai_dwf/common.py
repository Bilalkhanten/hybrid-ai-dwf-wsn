"""Shared helpers: configuration, seeding, run directories, metrics."""
from __future__ import annotations
import hashlib, json, os, platform, random, subprocess, sys
from pathlib import Path
import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent

STAGES = ["split", "mice", "smote", "init", "shuffle", "lgbm_bagging", "tpe", "adversarial", "bootstrap"]


def load_cfg(path: str | None = None) -> dict:
    p = Path(path) if path else PKG_ROOT / "config" / "default.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["_cfg_path"] = str(p)
    return cfg


def stage_seed(master: int, stage: str) -> int:
    """One deterministic seed per stage, derived from the master seed s0 (Table A3)."""
    h = hashlib.sha256(f"{master}:{stage}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


def all_stage_seeds(master: int) -> dict:
    return {s: stage_seed(master, s) for s in STAGES}


def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def run_dir(cfg: dict, dataset: str, seed: int) -> Path:
    d = Path(cfg["out_dir"]) / dataset / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def software_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ["tensorflow", "lightgbm", "optuna", "xgboost", "catboost", "sklearn", "imblearn", "numpy", "scipy", "pandas"]:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    try:
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except Exception:
        out["gpu"] = []
    return out


# ----------------------------------------------------------------------------- metrics
def confusion(y, p, n_classes):
    cm = np.bincount(y.astype(np.int64) * n_classes + p.astype(np.int64), minlength=n_classes ** 2)
    return cm.reshape(n_classes, n_classes)


def metrics_from_cm(cm: np.ndarray, benign: int | None = None, fpr_mode: str = "macro_ovr") -> dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        fpr_c = np.where(fp + tn > 0, fp / (fp + tn), 0.0)
    present = (cm.sum(1) > 0)
    if fpr_mode == "benign_vs_attack" and benign is not None:
        attack_rows = [i for i in range(cm.shape[0]) if i != benign]
        fpr = cm[benign, attack_rows].sum() / max(cm[benign].sum(), 1)  # benign flagged as any attack
    else:
        fpr = fpr_c[present].mean()
    return {
        "accuracy": 100 * tp.sum() / cm.sum(),
        "macro_f1": 100 * f1[present].mean(),
        "macro_precision": 100 * prec[present].mean(),
        "macro_recall": 100 * rec[present].mean(),
        "fpr": 100 * fpr,
    }


def evaluate(y, p, n_classes, benign=None, fpr_mode="macro_ovr") -> dict:
    return metrics_from_cm(confusion(y, p, n_classes), benign, fpr_mode)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
