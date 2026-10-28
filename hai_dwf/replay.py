"""Stage 2 of the adversarial protocol: problem-space replay (Section 3.11).

A feature-space adversarial vector is only a *feasible* attack if a packet sequence exists that (i) the attacker can generate,
(ii) keeps the malicious function, and (iii) yields the vector when passed through the REAL feature-extraction code.  This
module realises a target vector at packet level by constrained random search over packet mutations (padding, timing jitter,
fragmentation, interleaving with benign packets), re-extracts the features with the dataset's extractor, and reports

  valid_rate  = fraction of samples whose realised vector lies within `delta` (L_inf, scaled space) of the target AND passes
                the attack-function check,
  acc_replayed = ensemble accuracy on the REALISED vectors (not on the targets).

The extractor MUST be the code that produced the dataset's HFR features (e.g. CICFlowMeter for the CICIoT-2023 pcaps); it is
passed with --extractor module:function  (function(list[Pkt]) -> np.ndarray of the n_hfr RAW component values).
`attack_function_ok` defaults to conservative invariants and should be replaced by a sandboxed replay against a target
(Section 3.11) wherever one is available.
"""
from __future__ import annotations
import argparse, importlib
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Pkt:
    ts: float; length: int; proto: int = 6; sport: int = 0; dport: int = 0; flags: int = 0; src: str = ""; dst: str = ""; payload: int = 0


# ----------------------------------------------------------------------------- pcap I/O (scapy)
def read_pcap(path: str, limit: int | None = None) -> list[Pkt]:
    from scapy.all import rdpcap, IP, TCP, UDP
    out = []
    for i, p in enumerate(rdpcap(path)):
        if limit and i >= limit:
            break
        if IP not in p:
            continue
        ip = p[IP]; l4 = p[TCP] if TCP in p else (p[UDP] if UDP in p else None)
        out.append(Pkt(float(p.time), len(p), int(ip.proto), int(getattr(l4, "sport", 0)), int(getattr(l4, "dport", 0)),
                       int(getattr(l4, "flags", 0)), ip.src, ip.dst, len(bytes(l4.payload)) if l4 is not None else 0))
    return out


# ----------------------------------------------------------------------------- mutations (all attacker-feasible)
def pad(pk: list[Pkt], add: int, max_len: int = 1500) -> list[Pkt]:
    return [replace(p, length=min(max_len, p.length + add), payload=p.payload + max(0, min(add, max_len - p.length))) for p in pk]


def jitter(pk: list[Pkt], sigma: float, rng) -> list[Pkt]:
    ts = np.maximum.accumulate(np.array([p.ts for p in pk]) + rng.normal(0, sigma, len(pk)))
    return [replace(p, ts=float(t)) for p, t in zip(pk, ts)]


def fragment(pk: list[Pkt], mtu: int) -> list[Pkt]:
    out = []
    for p in pk:
        if p.length > mtu:
            n = int(np.ceil(p.length / mtu))
            out += [replace(p, ts=p.ts + k * 1e-6, length=min(mtu, p.length - k * mtu), payload=max(0, p.payload // n)) for k in range(n)]
        else:
            out.append(p)
    return out


def interleave(pk: list[Pkt], benign: list[Pkt], ratio: float, rng) -> list[Pkt]:
    n = int(ratio * len(pk))
    if n == 0 or not benign:
        return pk
    t0, t1 = pk[0].ts, pk[-1].ts
    extra = [replace(benign[i % len(benign)], ts=float(rng.uniform(t0, t1))) for i in range(n)]
    return sorted(pk + extra, key=lambda p: p.ts)


def apply_ops(pk, params, benign, rng):
    out = pad(pk, int(params["pad"])) if params["pad"] > 0 else pk
    if params["frag"] > 0:
        out = fragment(out, int(params["frag"]))
    if params["jit"] > 0:
        out = jitter(out, params["jit"], rng)
    return interleave(out, benign, params["inter"], rng)


def attack_function_ok(orig: list[Pkt], mut: list[Pkt], rate_floor: float = 0.5) -> bool:
    """Conservative default: the attack keeps targeting the same (dst, dport, proto) set, keeps >= rate_floor of the original
    packet rate to the victim, and the capture duration is within 25 %."""
    key = lambda p: (p.dst, p.dport, p.proto)
    o, m = {key(p) for p in orig}, {key(p) for p in mut if key(p) in {key(q) for q in orig}}
    if not o <= m:
        return False
    d0 = max(orig[-1].ts - orig[0].ts, 1e-9); d1 = mut[-1].ts - mut[0].ts
    r0 = len(orig) / d0; r1 = sum(1 for p in mut if key(p) in o) / max(d1, 1e-9)
    return r1 >= rate_floor * r0 and abs(d1 - d0) <= 0.25 * d0


# ----------------------------------------------------------------------------- realisation by search
SPACE = {"pad": (0, 400), "frag": (0, 0), "jit": (0.0, 0.05), "inter": (0.0, 1.0)}


def realise(pk, target_scaled, extractor, scale_fn, ctrl_mask, benign, rng, iters=60, fragments=(0, 64, 128, 256)):
    """Random local search over the mutation parameters minimising the controllable-component L_inf distance to the target."""
    best, bp, bd = None, None, np.inf
    for it in range(iters):
        params = {"pad": rng.uniform(*SPACE["pad"]), "frag": float(rng.choice(fragments)), "jit": rng.uniform(*SPACE["jit"]),
                  "inter": rng.uniform(*SPACE["inter"])}
        if bp is not None and it > iters // 2:   # local refinement
            params = {k: float(np.clip(bp[k] * rng.uniform(0.8, 1.2) + (rng.normal(0, 1e-3)), *SPACE[k])) if k != "frag" else bp[k] for k in bp}
        mut = apply_ops(pk, params, benign, rng)
        v = scale_fn(extractor(mut))
        d = float(np.abs((v - target_scaled) * ctrl_mask).max())
        if d < bd:
            best, bp, bd = (mut, v), params, d
    return best[0], best[1], bd, bp


def run_replay(ctx, windows_pk: list[list[Pkt]], targets_scaled, y: np.ndarray, extractor, delta: float, benign_pool,
               ctrl_mask, seed: int = 0, ok_fn=attack_function_ok, evade: bool = False, win_idx=None):
    """windows_pk[i]: original packets behind the LAST time step of window i; targets_scaled[i]: adversarial last-step vector.
    Returns validity and the accuracy of the ensemble on windows whose last step is replaced by the REALISED vector."""
    rng = np.random.default_rng(seed)
    scale_fn = lambda raw: np.clip((np.asarray(raw, np.float32) - ctx.pp.med) / ctx.pp.iqr, -1, 1)
    win_idx = np.arange(len(windows_pk)) if win_idx is None else np.asarray(win_idx)
    W = ctx.W0[win_idx].copy(); valid = []
    for i, pk in enumerate(windows_pk):
        if evade:   # packet-level mutation attack: validity = attack function preserved (no target vector to be close to)
            mut, v = mutate_evade(ctx, int(win_idx[i]), pk, extractor, benign_pool, rng); valid.append(ok_fn(pk, mut))
        else:
            mut, v, d, _ = realise(pk, targets_scaled[i], extractor, scale_fn, ctrl_mask, benign_pool, rng)
            valid.append(d <= delta and ok_fn(pk, mut))
        W[i, -1, :] = v
    valid = np.array(valid)
    pred = ctx.predict(W).argmax(1)
    yy = np.asarray(y)
    return {"valid_rate": float(valid.mean() * 100), "acc_replayed_all": float((pred == yy).mean() * 100),
            "acc_replayed_valid_only": float((pred[valid] == yy[valid]).mean() * 100) if valid.any() else float("nan"), "n": int(len(yy))}


def mutate_evade(ctx, i, pk, extractor, benign, rng, iters=80):
    """Packet-level mutation attack with NO target vector: random search over mutations minimising the ensemble's probability of
    the true class (attacker queries the detector).  Returns (mutated packets, realised scaled vector)."""
    scale_fn = lambda raw: np.clip((np.asarray(raw, np.float32) - ctx.pp.med) / ctx.pp.iqr, -1, 1)
    best, bp = None, np.inf
    for _ in range(iters):
        params = {"pad": rng.uniform(*SPACE["pad"]), "frag": float(rng.choice((0, 64, 128, 256))), "jit": rng.uniform(*SPACE["jit"]),
                  "inter": rng.uniform(*SPACE["inter"])}
        mut = apply_ops(pk, params, benign, rng); v = scale_fn(extractor(mut))
        W = ctx.W0[i:i + 1].copy(); W[0, -1, :] = v
        p = float(ctx.predict(W)[0, ctx.y[i]])
        if p < bp:
            best, bp = (mut, v), p
    return best


def load_callable(spec: str):
    mod, fn = spec.split(":"); return getattr(importlib.import_module(mod), fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg"); ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--attack", default="PGD", choices=["FGSM", "PGD", "DeepFool", "CW", "BlackBox", "TreeEvasion", "GateManip", "Mutation"]); ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--pcap", required=True, help="pcap(s) of the attack traffic behind the sampled windows, one window per file or a manifest csv")
    ap.add_argument("--manifest", required=True, help="csv: window_index,pcap_path,start_pkt,end_pkt  (links adversarial windows to packets)")
    ap.add_argument("--extractor", required=True, help="module:function mapping list[Pkt] -> raw HFR vector")
    ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    import pandas as pd
    from .common import load_cfg, write_json
    from . import adversarial as A
    cfg = load_cfg(a.cfg); ctx = A.Ctx(cfg, a.dataset, a.seed); eps = a.eps or cfg["adversarial"]["report_eps"]
    evade = a.attack == "Mutation"
    fn = {"Mutation": lambda: None, "FGSM": lambda: A.fgsm(ctx, eps, True), "PGD": lambda: A.pgd(ctx, eps, True, cfg["adversarial"]["pgd_steps"]),
          "BlackBox": lambda: A.square_blackbox(ctx, eps, True, cfg["adversarial"]["query_budget"]), "TreeEvasion": lambda: A.tree_evasion(ctx, eps, True),
          "GateManip": lambda: A.gate_manipulation(ctx, eps, True, cfg["adversarial"]["pgd_steps"]),
          "DeepFool": lambda: A.deepfool(ctx, eps, True), "CW": lambda: A.cw_l2(ctx, eps, True)}[a.attack]
    Wa = fn(); man = pd.read_csv(a.manifest); ext = load_callable(a.extractor)
    pk_all = {p: read_pcap(p) for p in man.pcap_path.unique()}
    wins = [pk_all[r.pcap_path][int(r.start_pkt):int(r.end_pkt)] for r in man.itertuples()]
    idx = man.window_index.to_numpy()
    benign = [p for pk in pk_all.values() for p in pk][:2000]
    out = run_replay(ctx, wins, None if evade else Wa[idx, -1, :], ctx.y[idx], ext, a.delta, benign, ctx.mask, a.seed, evade=evade, win_idx=idx)
    write_json(ctx.rd / f"replay_{a.attack}.json", {"attack": a.attack, "eps": eps, "delta": a.delta, **out})
    print(out)


if __name__ == "__main__":
    main()
