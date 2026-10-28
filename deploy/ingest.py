"""Ingest REAL board measurements (Section 3.12) and produce deploy/measurements.json for Table 20.

Nothing here estimates or simulates: every number comes from files you record on the target board.  Expected inputs per board
(board names: m55, u74; optional tier boards: telosb, z1, remote, nrf52840, stm32wb55):

  <dir>/stages_<board>.csv   decision_id,stage,cycles,cpu_hz       stage in {capture,hfr,scale,cnn,bilstm,lgbm,dwf,total}
                             (cycle counters: DWT CYCCNT on Cortex-M55, rdcycle on U74; >= 10,000 decisions)
  <dir>/energy_<board>.csv   time_s,current_uA,stage_id            current-profiler export (e.g. Otii/PPK2), stage_id from GPIO
                             markers (0 = idle, 1..7 = stages in the order above); use --vdd and --idle-uA
  <dir>/memory_<board>.json  {"flash_kb": {"cnn":..,"bilstm":..,"lgbm":..,"dwf":..,"total":..},
                              "sram_peak_kb": {"<stage>":..,"total":..}, "numeric_format": {"cnn":"int8",...}}
                             (from the linker map and stack/arena high-water marks; see firmware/bench.c)
  <dir>/radio_<role>.csv     window_id,tx_bytes,latency_ms,retx,energy_uJ    role in {leaf, relay} (nRF52840-DK / RE-Mote)
  <dir>/e2e_<board>.csv      window_id,t_capture_ms,t_decision_ms,window_rate_hz
  <dir>/power_<board>.csv    rate_hz,mean_power_uW                           idle and sleep current subtracted
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

STAGES = ["capture", "hfr", "scale", "cnn", "bilstm", "lgbm", "dwf"]


def stage_table(d: Path, board: str, vdd: float, idle_uA: float, min_decisions: int):
    out = {}
    p = d / f"stages_{board}.csv"
    if p.exists():
        df = pd.read_csv(p)
        n = df.decision_id.nunique()
        if n < min_decisions:
            print(f"[warn] {p}: only {n} decisions (< {min_decisions})", file=sys.stderr)
        df["ms"] = df.cycles / df.cpu_hz * 1e3
        for st, g in df.groupby("stage"):
            out.setdefault(st, {})["latency_ms"] = {"mean": g.ms.mean(), "sd": g.ms.std(ddof=1), "n": int(len(g))}
    p = d / f"energy_{board}.csv"
    if p.exists():
        e = pd.read_csv(p)
        dt = np.diff(e.time_s.to_numpy(), append=e.time_s.iloc[-1])
        uj = vdd * np.maximum(e.current_uA.to_numpy() - idle_uA, 0) * dt            # V * uA * s = uJ
        e["uJ"] = uj
        e["seg"] = (e.stage_id != e.stage_id.shift()).cumsum()
        per = e[e.stage_id > 0].groupby(["seg", "stage_id"]).uJ.sum().reset_index()
        for sid, g in per.groupby("stage_id"):
            st = STAGES[int(sid) - 1]
            out.setdefault(st, {})["energy_uJ"] = {"mean": g.uJ.mean(), "sd": g.uJ.std(ddof=1), "n": int(len(g))}
    p = d / f"memory_{board}.json"
    if p.exists():
        m = json.loads(p.read_text()); out["_memory"] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--boards", nargs="+", default=["m55", "u74"])
    ap.add_argument("--vdd", type=float, default=3.3); ap.add_argument("--idle-uA", type=float, default=0.0)
    ap.add_argument("--min-decisions", type=int, default=10000); ap.add_argument("--out", default="deploy/measurements.json")
    a = ap.parse_args(); d = Path(a.dir)
    res = {"boards": {b: stage_table(d, b, a.vdd, a.idle_uA, a.min_decisions) for b in a.boards}, "radio": {}, "e2e": {}, "power": {}}
    for role in ("leaf", "relay"):
        p = d / f"radio_{role}.csv"
        if p.exists():
            r = pd.read_csv(p)
            res["radio"][role] = {"payload_bytes": float(r.tx_bytes.mean()), "latency_ms": {"mean": r.latency_ms.mean(), "sd": r.latency_ms.std(ddof=1)},
                                  "energy_uJ": {"mean": r.energy_uJ.mean(), "sd": r.energy_uJ.std(ddof=1)}, "retx_mean": float(r.retx.mean()), "n": int(len(r))}
    for b in a.boards:
        p = d / f"e2e_{b}.csv"
        if p.exists():
            r = pd.read_csv(p); lat = r.t_decision_ms - r.t_capture_ms
            res["e2e"][b] = {"latency_ms": {"mean": lat.mean(), "sd": lat.std(ddof=1)}, "window_rate_hz": float(r.window_rate_hz.mean()), "n": int(len(r))}
        p = d / f"power_{b}.csv"
        if p.exists():
            r = pd.read_csv(p); res["power"][b] = {"rows": r.to_dict("records")}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float)); print("wrote", a.out)


if __name__ == "__main__":
    main()
