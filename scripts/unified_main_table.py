# -*- coding: utf-8 -*-
"""UNIFIED MAIN TABLE — one protocol for every row.

Protocol (fixed; nothing is chosen per row):
  private set   data/coco_private/coco_layouts.jsonl, N = 2859 (whole file)
  public pool   instances_train2017, max 30 sizes/class, K = 1000 candidates,
                seed 0
  budget        eps = 5, delta = 1e-5, T = 1, sigma = 1.0545 (RDP)
  support       per-arm nearest-candidate vote histogram
  release       SAME for every arm: 1110 with-replacement draws from the
                support-weighted candidate distribution (decoupled release),
                then the S4 class correction fed the POST-NOISE support,
                t = 0.7, seed 3
  replication   5 DP noise seeds x 12 inner release resamples
                (inner resamples only reduce release variance; they are not
                 extra DP seeds)
  metrics       six normalised errors against the public prior (=1.0) plus,
                clearly labelled as AUXILIARY, the mean of the four geometric
                ones (Rel/Ctr/Size/Scale)

Why the decoupled release for every row: it makes the release policy identical
across arms, so a row difference can only come from the support geometry. The
original (evolve-based) release is a separate ablation and is not mixed in.

Rows: Public-only, Flat, Sorted-Flat, Chamfer, LTSim, Chamfer+Card,
      LTSim+Card, CA-PE (DP), CA-PE (no noise)
"""
import io, os, sys, json, random, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import importlib.util

from layout.distance import DistanceConfig, layout_distance
from layout.ltsim import ltsim_distance
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco
from evaluation.layout_metrics import layout_set_report

NAMES = ["Count", "Class", "Rel", "Ctr", "Size", "Scale"]
KEYS = ["count_TV", "class_TV", "relation_marginal_TV", "center_TV",
        "size_TV", "scale_W1"]
PRIOR = {"count_TV": 0.03253147953830014, "class_TV": 0.1426724971660939,
         "relation_marginal_TV": 0.14722313485315913,
         "center_TV": 0.12377635357751908, "size_TV": 0.15418084244481628,
         "scale_W1": 0.10755081712986059}
GEO = ["Rel", "Ctr", "Size", "Scale"]
REPS = 12
NSEED = 5
# seed list and output path are overridable so the same protocol can be rerun
# with more outer DP-noise seeds (runs/SEED_PRE_REGISTRATION.md). Defaults
# reproduce the frozen table exactly.
NOISE_SEEDS = list(range(NSEED))
OUT_PATH = os.path.join(ROOT, "runs", "UNIFIED_TABLE.json")
PROG = os.path.join(ROOT, "runs", "unified_progress.txt")


def log(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = [name]
    spec.loader.exec_module(m)
    return m


def main():
    global NOISE_SEEDS, OUT_PATH, PROG
    if len(sys.argv) > 1 and sys.argv[1] == "--seeds":
        NOISE_SEEDS = [int(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 3 and sys.argv[3] == "--out":
        OUT_PATH = os.path.join(ROOT, sys.argv[4]) if not os.path.isabs(sys.argv[4]) else sys.argv[4]
        PROG = OUT_PATH.replace(".json", "_progress.txt")
    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"),
                           encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"),
                 encoding="utf-8")]
    assert len(priv) == 2859, len(priv)
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    cands = seed_candidates(pool, 1000, MutationConfig(rng=random.Random(0)))
    K = len(cands)
    nobj = np.array([len(C) for C in cands], dtype=float)
    log(f"priv={len(priv)} cands={K}")

    # ---------- distance matrices, cached ----------
    def get_matrix(tag, fn):
        p = os.path.join(ROOT, "runs", f"dist_{tag}.npy")
        if os.path.exists(p):
            log(f"  matrix {tag}: cached")
            return np.load(p)
        t0 = time.time()
        D = np.zeros((len(priv), K), dtype=np.float32)
        for i, L in enumerate(priv):
            for j, C in enumerate(cands):
                D[i, j] = fn(L, C)
            if (i + 1) % 500 == 0:
                log(f"  matrix {tag}: {i+1}/{len(priv)} ({time.time()-t0:.0f}s)")
        np.save(p, D)
        log(f"  matrix {tag}: done in {time.time()-t0:.0f}s")
        return D

    flat_m = load_module(os.path.join(ROOT, "scripts", "run_flat_tabpe.py"), "flatmod")
    sorted_m = load_module(os.path.join(ROOT, "scripts", "run_sorted_flat_tabpe.py"), "sortedmod")
    cls_to_idx = {c: i for i, c in enumerate(vocab)}

    arms = {}
    arms["Flat"] = ("matrix", flat_m.flat_votes_matrix(priv, cands, cls_to_idx), 0.0, 0.0)
    arms["Sorted-Flat"] = ("matrix", sorted_m.flat_votes_matrix(priv, cands, cls_to_idx), 0.0, 0.0)
    D_ch = get_matrix("chamfer", lambda L, C: layout_distance(
        L, C, DistanceConfig(matching=MatchingConfig(method="chamfer"),
                             gamma=0.0))["total"])
    arms["Chamfer"] = ("matrix", D_ch, 0.0, 0.0)
    arms["Chamfer+Card"] = ("matrix", D_ch, 2.5, 0.5)
    D_lt = get_matrix("ltsim_plain", lambda L, C: ltsim_distance(L, C))
    arms["LTSim"] = ("matrix", D_lt, 0.0, 0.0)
    arms["LTSim+Card"] = ("matrix", D_lt, 2.5, 0.5)
    v = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))
    arms["CA-PE (DP)"] = ("votes", v["ca"], 0.0, 0.0)
    arms["CA-PE (no noise)"] = ("votes", v["ca"], 0.0, 0.0)

    def votes_from_matrix(D, mu, gamma):
        out = np.zeros(len(priv), dtype=np.int32)
        for i in range(len(priv)):
            m = len(priv[i].objects)
            d = D[i].astype(np.float64)
            if mu > 0:
                d = d + mu * np.abs(m - nobj) / m
            if gamma > 0:
                d = d + gamma * np.abs(m - nobj) / np.maximum(m, nobj)
            out[i] = int(np.argmin(d))
        return out

    hist = {}
    for name, (kind, payload, mu, gamma) in arms.items():
        if kind == "matrix":
            a = votes_from_matrix(payload, mu, gamma)
            hist[name] = np.bincount(a, minlength=K).astype(float)
        else:
            hist[name] = payload.astype(float)
        log(f"  votes {name}: nnz={int((hist[name]>0).sum())}")

    # ---------- release + metrics ----------
    sigma = sigma_for_epsilon(5.0, 1e-5, 1)
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    fv = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(fv)
    M = fv.M

    out = {}
    rows = [("Public-only", np.ones(K))]
    rows += [(n, hist[n]) for n in ("Flat", "Sorted-Flat", "Chamfer", "LTSim",
                                    "Chamfer+Card", "LTSim+Card",
                                    "CA-PE (DP)", "CA-PE (no noise)")]
    for name, h in rows:
        t0 = time.time()
        per_seed = []
        for ns in NOISE_SEEDS:
            if name != "Public-only" and name != "CA-PE (no noise)":
                noisy = np.clip(h + np.random.default_rng(3000 + ns)
                                .normal(0, sigma, K), 0, None)
            else:
                noisy = h.astype(float).copy()
            w = noisy / noisy.sum()
            vals = []
            for rs in range(REPS):
                rel = [cands[j] for j in random.Random(500 + rs).choices(
                    range(K), weights=w.tolist(), k=M)]
                o = fv.s4(rel, noisy, cands, 0.7, 3)
                rep = layout_set_report(priv, o, vocab)
                vals.append({n: rep[k] / PRIOR[k] for n, k in zip(NAMES, KEYS)})
            per_seed.append({n: float(np.mean([x[n] for x in vals])) for n in NAMES})
        mean = {n: float(np.mean([s[n] for s in per_seed])) for n in NAMES}
        sd = {n: float(np.std([s[n] for s in per_seed], ddof=1)) for n in NAMES}
        avg = float(np.mean([mean[n] for n in NAMES]))
        geo = float(np.mean([mean[n] for n in GEO]))
        out[name] = {"mean": mean, "seed_sd": sd, "Avg": avg, "Geo4": geo,
                     "per_seed": per_seed}
        log(f"DONE {name:16s} Avg={avg:.4f} Geo4={geo:.4f}  "
            + " ".join(f"{n}={mean[n]:.3f}" for n in NAMES)
            + f"  ({time.time()-t0:.0f}s)")

    json.dump({"protocol": {"eps": 5.0, "delta": 1e-5, "T": 1, "sigma": sigma,
                            "release": "decoupled (support-weighted resample)",
                            "s4": {"support": "post-noise", "t": 0.7, "seed": 3},
                            "seeds": len(NOISE_SEEDS),
                            "noise_seeds": NOISE_SEEDS,
                            "inner_resamples": REPS,
                            "release_size": M, "candidates": K},
               "rows": out},
              open(OUT_PATH, "w"), indent=1)
    log("ALL-DONE")


if __name__ == "__main__":
    main()
