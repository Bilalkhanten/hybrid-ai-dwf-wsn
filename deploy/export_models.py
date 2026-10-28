"""Export the trained components for the target boards and report the artefact sizes.

    python deploy/export_models.py --cfg config/default.yaml --dataset ciciot2023 --seed 0 --out deploy/export

* CNN / BiLSTM -> TensorFlow Lite int8 (post-training quantisation with a representative set from the validation partition).
  Build with TFLite-Micro (+ Ethos-U vela compile for the Cortex-M55 + Ethos-U55) or any RISC-V runtime for the U74.
* LightGBM -> self-contained C (if/else trees, float32 thresholds) usable without a runtime.
* hfr_extract.h : the 28 x 10 histogram entropy + scaling constants (median, IQR, fitting-partition min/max) so that feature
  extraction runs ON the device with the same numbers as the host pipeline.
Flash/SRAM of the final firmware must come from the linker map of the real build (see firmware/bench.c), not from these file sizes.
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hai_dwf.common import load_cfg, run_dir, read_json, write_json


def lgbm_to_c(booster, n_classes, path):
    dump = booster.dump_model(); trees = dump["tree_info"]
    lines = ["#include <stdint.h>", f"#define N_CLASSES {n_classes}", f"#define N_TREES {len(trees)}"]
    def emit(node, ind):
        if "leaf_value" in node:
            return [f"{ind}return {node['leaf_value']:.9g}f;"]
        f, th = node["split_feature"], node["threshold"]
        l = [f"{ind}if (x[{f}] <= {float(th):.9g}f) {{"] + emit(node["left_child"], ind + "  ") + [f"{ind}}} else {{"] + emit(node["right_child"], ind + "  ") + [f"{ind}}}"]
        return l
    for i, t in enumerate(trees):
        lines += [f"static float tree{i}(const float *x) {{"] + emit(t["tree_structure"], "  ") + ["}"]
    lines += ["void lgbm_raw(const float *x, float *out) {", "  for (int c = 0; c < N_CLASSES; ++c) out[c] = 0.f;"]
    for i in range(len(trees)):
        lines.append(f"  out[{i % n_classes}] += tree{i}(x);")
    lines += ["}"]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg"); ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="deploy/export")
    a = ap.parse_args(); cfg = load_cfg(a.cfg); rd = run_dir(cfg, a.dataset, a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf, lightgbm as lgb
    from hai_dwf import data as D
    from hai_dwf.models import TemporalAttention  # noqa
    pp_ = pickle.load(open(rd / "preproc.pkl", "rb")); pp = pp_["pp"]
    P = np.load(rd / "partitions.npz"); T = cfg["window"]
    sizes = {}
    # representative data: windows from the validation partition
    df = D.load_raw(cfg, a.dataset); rows = np.flatnonzero(P["part"] == 1)
    Z = pp.transform(D.apply_hfr(df.iloc[rows], pp_["spec"])); idx = D.window_index(P["block"][rows], T)
    sel = np.random.default_rng(0).permutation(len(rows))[:500]; rep = Z[idx[sel]].astype(np.float32)
    for name, shape_fn in (("cnn", lambda r: r[:, -1, :]), ("bilstm", lambda r: r)):
        m = tf.keras.models.load_model(rd / f"{name}.keras")
        conv = tf.lite.TFLiteConverter.from_keras_model(m); conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ((shape_fn(rep[i:i + 1]),) for i in range(len(rep)))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        try:
            blob = conv.convert()
        except Exception as e:          # LSTM int8 conversion is toolchain dependent: fall back to dynamic-range and say so
            print(f"[warn] full int8 failed for {name}: {e}; using dynamic-range quantisation")
            conv = tf.lite.TFLiteConverter.from_keras_model(m); conv.optimizations = [tf.lite.Optimize.DEFAULT]; blob = conv.convert()
        (out / f"{name}.tflite").write_bytes(blob); sizes[f"{name}_tflite_kb"] = len(blob) / 1024
    gb = lgb.Booster(model_file=str(rd / "lgbm.txt")); lgbm_to_c(gb, int(P["y"].max()) + 1, out / "lgbm_model.c")
    sizes["lgbm_c_source_kb"] = (out / "lgbm_model.c").stat().st_size / 1024
    h = ["#pragma once", "/* generated: constants of the host pipeline, copy to firmware */", f"#define HFR_N {Z.shape[1]}", f"#define WIN_T {T}", f"#define HIST_BINS {cfg['bins']}"]
    for nm, arr in (("HFR_MED", pp.med), ("HFR_IQR", pp.iqr), ("HFR_MIN", pp.mn), ("HFR_MAX", pp.mx)):
        h.append(f"static const float {nm}[{len(arr)}] = {{" + ",".join(f"{v:.9g}f" for v in arr) + "};")
    (out / "hfr_extract.h").write_text("\n".join(h))
    write_json(out / "artefact_sizes.json", sizes); print(sizes)


if __name__ == "__main__":
    main()
