# -*- coding: utf-8 -*-
"""Class-term weight sweep: Avg (distribution fidelity) vs per-record class TV
(semantic correspondence).

Addresses the reviewer objection "removing the class term improves Avg, so the
component is useless": report both objectives and show the Pareto trade-off
instead of a single ablation point.

Protocol identical to scripts/ablation_noisy.py:
  T=1, eps=5, delta=1e-5, sigma=1.0545, K=1000 (seed 0), M=1110,
  S4(t=0.7, seed 3), 10 outer noise seeds x 12 release resamples.
lambda_class in {0, 0.5, 1, 2}; everything else fixed (ctr=2, size=1, giou=1,
mu=2.5, gamma=0.5, Hungarian).

The correspondence metric: top-8 private classes; for every private layout, the
TV between its class histogram and that of its nearest candidate under the same
cost; averaged over all 2859 records (deterministic, no noise).
"""
import os, sys, json, random, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import importlib.util
from collections import Counter

from layout.distance import DistanceConfig, layout_distance
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco
from evaluation.layout_metrics import layout_set_report

OUT = os.path.join(ROOT, "runs")
LOG = os.path.join(OUT, "lambda_class_sweep.log")
NAMES = ["Count", "Class", "Rel", "Ctr", "Size", "Scale"]
KEYS = ["count_TV", "class_TV", "relation_marginal_TV", "center_TV",
        "size_TV", "scale_W1"]
PRIOR = {"count_TV": 0.03253147953830014, "class_TV": 0.1426724971660939,
         "relation_marginal_TV": 0.14722313485315913,
         "center_TV": 0.12377635357751908, "size_TV": 0.15418084244481628,
         "scale_W1": 0.10755081712986059}
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
REPS, NSEED, M, K = 12, 10, 1110, 1000

_W = {}


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def class_hist(lay, idx, n):
    h = np.zeros(n)
    for o in lay.objects:
        k = idx.get(o.cls)
        if k is not None:
            h[k] += 1.0
    s = h.sum()
    return h / s if s else h


def _init(cands, cfg, idx, n_cls):
    _W["c"] = cands
    _W["cfg"] = cfg
    _W["idx"] = idx
    _W["n"] = n_cls
    _W["cache"] = [class_hist(c, idx, n_cls) for c in cands]


def _chunk(privs):
    from layout.distance import layout_distance
    votes = np.zeros(len(_W["c"]))
    tv_sum = 0.0
    for L in privs:
        hl = class_hist(L, _W["idx"], _W["n"])
        best_j, best_d = 0, float("inf")
        for j, C in enumerate(_W["c"]):
            d = layout_distance(L, C, _W["cfg"])["total"]
            if d < best_d:
                best_d, best_j = d, j
        votes[best_j] += 1.0
        tv_sum += float(np.abs(hl - _W["cache"][best_j]).sum() / 2.0)
    return votes, tv_sum, len(privs)


def score(priv, cands, cfg, idx, n_cls, jobs=6):
    import multiprocessing as mp
    n_chunks = max(jobs, min(8 * jobs, len(priv)))
    size = (len(priv) + n_chunks - 1) // n_chunks
    chunks = [priv[i:i + size] for i in range(0, len(priv), size)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_init,
                  initargs=(list(cands), cfg, idx, n_cls)) as p:
        parts = p.map(_chunk, chunks)
    v = np.zeros(len(cands))
    tv = 0.0
    tot = 0
    for a, b, c in parts:
        v += a
        tv += b
        tot += c
    return v, tv / max(tot, 1)


def main():
    open(LOG, "w", encoding="utf-8").close()
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    fv = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(fv)

    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"), encoding="utf-8")]
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    cands = seed_candidates(pool, K, MutationConfig(rng=random.Random(0)))
    freq = Counter(o.cls for L in priv for o in L.objects)
    top8 = [c for c, _ in freq.most_common(8)]
    idx = {c: i for i, c in enumerate(top8)}
    sigma = sigma_for_epsilon(5.0, 1e-5, 1)
    log(f"priv={len(priv)} cands={len(cands)} sigma={sigma:.4f}")

    results = []
    for lc in LAMBDAS:
        t0 = time.time()
        cfg = DistanceConfig(matching=MatchingConfig(lambda_class=lc,
                                                    unmatched_penalty=2.5),
                             gamma=0.5)
        vp = os.path.join(OUT, f"lambda_votes_{lc}.npz")
        if os.path.exists(vp):
            z = np.load(vp)
            h, ctv = z["h"].astype(float), float(z["ctv"])
            log(f"[lc={lc}] votes cached nnz={int((h>0).sum())} ctv={ctv:.4f}")
        else:
            h, ctv = score(priv, cands, cfg, idx, len(top8))
            np.savez(vp, h=h, ctv=ctv)
            log(f"[lc={lc}] votes nnz={int((h>0).sum())} classTV={ctv:.4f} "
                f"({time.time()-t0:.0f}s)")

        per = []
        for ns in range(NSEED):
            noisy = np.clip(h + np.random.default_rng(3000 + ns).normal(0, sigma, K), 0, None)
            w = noisy / noisy.sum()
            vals = []
            for rs in range(REPS):
                rel = [cands[j] for j in random.Random(500 + rs).choices(
                    range(K), weights=w.tolist(), k=M)]
                o = fv.s4(rel, noisy, cands, 0.7, 3)
                rep = layout_set_report(priv, o, vocab)
                vals.append({n: rep[k] / PRIOR[k] for n, k in zip(NAMES, KEYS)})
            per.append({n: float(np.mean([x[n] for x in vals])) for n in NAMES})
        mean = {n: float(np.mean([s[n] for s in per])) for n in NAMES}
        sd = {n: float(np.std([s[n] for s in per], ddof=1)) for n in NAMES}
        rec = {"lambda_class": lc, "class_tv_per_record": ctv,
               "mean": mean, "seed_sd": sd,
               "Avg": float(np.mean(list(mean.values()))),
               "per_seed": per}
        results.append(rec)
        log(f"[lc={lc}] Avg={rec['Avg']:.4f} classTV={ctv:.4f} "
            f"total {time.time()-t0:.0f}s")

    json.dump(results, open(os.path.join(OUT, "LAMBDA_CLASS_SWEEP.json"), "w"), indent=1)
    print("\n=== lambda_class sweep ===")
    print(f"{'lambda':>8}{'Avg':>9}{'per-rec class TV':>18}")
    for r in results:
        print(f"{r['lambda_class']:>8}{r['Avg']:>9.4f}{r['class_tv_per_record']:>18.4f}")
    log("ALL-DONE")


if __name__ == "__main__":
    main()
