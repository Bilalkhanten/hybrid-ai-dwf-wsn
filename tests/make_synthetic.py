"""SYNTHETIC data for exercising the code path only (never use any number derived from it in the paper)."""
import numpy as np, pandas as pd, sys
from pathlib import Path

def make(name, n, n_feat, n_cls, seed, wsn=False):
    rng = np.random.default_rng(seed)
    rows, t = [], 0
    names = ["pkt_len", "iat", "bytes_per_s", "pkts_count", "flag_syn", "duration"] + (["lqi_mean_dev", "hop_count", "routing_changes"] if wsn else []) + \
            [f"f{i}" for i in range(n_feat - 6 - (3 if wsn else 0))]
    sess = 0
    while len(rows) < n:
        c = int(rng.integers(0, n_cls)); L = int(rng.integers(30, 80)); mu = rng.normal(0, 1, len(names)) * (0.4 + c)
        spread = 0.3 + 0.35 * c
        for _ in range(L):
            x = np.abs(mu + rng.normal(0, spread, len(names)))
            rows.append([sess, t] + list(x) + [("BENIGN" if c == 0 else f"ATK{c}")]); t += 1
        sess += 1
    df = pd.DataFrame(rows[:n], columns=["sess", "t"] + names + ["label"])
    df.loc[rng.random(len(df)) < 0.01, "f0"] = np.nan if "f0" in df else np.nan
    return df

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data"); out.mkdir(exist_ok=True, parents=True)
    spec = {"ciciot2023": (9000, 40, 5, False), "edge_iiotset": (6000, 30, 4, False), "nf_ton_iot_v2": (6000, 12, 4, False), "wsn_ds": (5000, 19, 4, True)}
    for i, (k, (n, f, c, w)) in enumerate(spec.items()):
        make(k, n, f, c, i, w).to_csv(out / f"smoke_{k}.csv", index=False)
    print("written", list(spec))
