#!/usr/bin/env python3
"""Run the full validation pipeline.  Each (dataset, seed) job is a separate process (clean GPU memory, resumable: finished jobs are skipped).

    python run_all.py --cfg config/default.yaml                      # everything
    python run_all.py --cfg config/default.yaml --stages components fusion
    python run_all.py --cfg config/default.yaml --datasets ciciot2023 --seeds 0 1 2
"""
import argparse, subprocess, sys
from hai_dwf.common import load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="config/default.yaml")
ap.add_argument("--stages", nargs="+", default=["components", "fusion", "transfer", "adversarial", "aggregate"])
ap.add_argument("--datasets", nargs="+"); ap.add_argument("--seeds", nargs="+", type=int)
ap.add_argument("--adv-seeds", nargs="+", type=int, help="seeds for the (expensive) adversarial stage; default: all seeds")
ap.add_argument("--force", action="store_true")
a = ap.parse_args(); cfg = load_cfg(a.cfg)
dss = a.datasets or cfg["dataset_order"]; seeds = a.seeds or cfg["seeds"]
run = lambda *args: subprocess.run([sys.executable, "-m", *args, "--cfg", a.cfg] + (["--force"] if a.force and "aggregate" not in args[0] else []), check=True)
if "components" in a.stages:
    for s in seeds:
        for d in dss:
            run("hai_dwf.components", "--dataset", d, "--seed", str(s))
if "fusion" in a.stages:
    for s in seeds:
        for d in dss:
            run("hai_dwf.fusion", "--dataset", d, "--seed", str(s))
if "transfer" in a.stages:                      # needs all datasets of a seed to be finished
    for s in seeds:
        run("hai_dwf.transfer", "--seed", str(s))
if "adversarial" in a.stages:
    for s in (a.adv_seeds or seeds):
        run("hai_dwf.adversarial", "--dataset", cfg["dataset_order"][0], "--seed", str(s))
if "aggregate" in a.stages:
    subprocess.run([sys.executable, "-m", "hai_dwf.aggregate", "--cfg", a.cfg, "--out", "results"], check=True)
