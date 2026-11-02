# -*- coding: utf-8 -*-
"""Probe: T=12 CA-PE with the class term removed (lambda_class = 0).

Question: does dropping d_cls make multi-round evolution degrade (the claim
"without the class term multi-round collapses")? At T=1 the same ablation is
slightly BETTER on Avg (0.6017 vs 0.6360) and its votes are more dispersed
(ESS 308.1 vs 273.7), so the claim is unsupported there; this run gives the
first multi-round data point.

Protocol copied from scripts/final_validation.py (frozen):
  T=12, eps=5, delta=1e-5, sigma from the RDP accountant, K=1000 candidates,
  decoupled release of M=1110 by the final support, S4(t=0.7, seed=3).
Only lambda_class changes.

Per round it also records vote concentration and the mean per-record class TV
between each private layout and its chosen candidate (top-8 classes), which is
the quantity the class term is supposed to protect.
"""
import os, sys, json, random, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from collections import Counter
import importlib.util

from layout.distance import DistanceConfig, layout_distance
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco
from pe.selection import DiversitySelector, SelectionConfig, prior_strata_quotas

OUT = os.path.join(ROOT, "runs", "T12_NOCLASS_PROBE.json")
LOG = os.path.join(ROOT, "runs", "t12_noclass_probe.log")
K, PARENTS, FRESH, M = 1000, 280, 550, 1110
T = int(os.environ.get("PROBE_T", "12"))
N_PRIV = int(os.environ.get("PROBE_N", "2859"))
DELTA = 1e-5


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def load_fv():
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(m)
    return m


def class_hist(layout, idx, n):
    h = np.zeros(n)
    for o in layout.objects:
        k = idx.get(o.cls)
        if k is not None:
            h[k] += 1.0
    s = h.sum()
    return h / s if s else h


_W = {}


def _init_worker(cands, cfg, idx, n_cls):
    _W["cands"] = cands
    _W["cfg"] = cfg
    _W["cache"] = [class_hist(c, idx, n_cls) for c in cands]
    _W["idx"] = idx
    _W["n"] = n_cls


def _vote_chunk(privs):
    votes = np.zeros(len(_W["cands"]))
    tv_sum = 0.0
    for L in privs:
        hl = class_hist(L, _W["idx"], _W["n"])
        best_j, best_d = 0, float("inf")
        for j, C in enumerate(_W["cands"]):
            d = layout_distance(L, C, _W["cfg"])["total"]
            if d < best_d:
                best_d, best_j = d, j
        votes[best_j] += 1.0
        tv_sum += float(np.abs(hl - _W["cache"][best_j]).sum() / 2.0)
    return votes, tv_sum, len(privs)


def score_round(priv, cands, cfg, idx, n_cls, tag=""):
    """Votes + mean per-record class TV, one pass over the candidate pool."""
    jobs = int(os.environ.get("PROBE_JOBS", "1"))
    if jobs > 1:
        import multiprocessing as mp
        n_chunks = max(jobs, min(8 * jobs, len(priv)))
        size = (len(priv) + n_chunks - 1) // n_chunks
        chunks = [priv[i:i + size] for i in range(0, len(priv), size)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs, initializer=_init_worker,
                      initargs=(list(cands), cfg, idx, n_cls)) as pool:
            parts = pool.map(_vote_chunk, chunks)
        votes = np.zeros(len(cands))
        tv_sum = 0.0
        total = 0
        for v, s, k in parts:
            votes += v
            tv_sum += s
            total += k
        return votes, tv_sum / max(total, 1)

    votes = np.zeros(len(cands))
    tv_sum = 0.0
    cache = [class_hist(c, idx, n_cls) for c in cands]
    for i, L in enumerate(priv):
        hl = class_hist(L, idx, n_cls)
        best_j, best_d = 0, float("inf")
        for j, C in enumerate(cands):
            d = layout_distance(L, C, cfg)["total"]
            if d < best_d:
                best_d, best_j = d, j
        votes[best_j] += 1.0
        tv_sum += float(np.abs(hl - cache[best_j]).sum() / 2.0)
        if (i + 1) % 500 == 0:
            log(f"  {tag} scored {i+1}/{len(priv)}")
    return votes, tv_sum / max(len(priv), 1)


def main():
    open(LOG, "w", encoding="utf-8").close()
    fv = load_fv()
    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"),
                           encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"),
                 encoding="utf-8")][:N_PRIV]
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    ff = {c: x / max(sum(pool.class_freq.values()), 1) for c, x in pool.class_freq.items()}
    strata = prior_strata_quotas(pool, freq=ff)
    sel = DiversitySelector(SelectionConfig(
        n_select=PARENTS, n_elite=PARENTS // 3, lambda_redundancy=0.6,
        min_bucket_frac=0.12, rare_frac=0.30, strata_quotas=strata, freq=ff))

    # top-8 private classes, as in the matching ablation
    freq = Counter(o.cls for L in priv for o in L.objects)
    top8 = [c for c, _ in freq.most_common(8)]
    idx = {c: i for i, c in enumerate(top8)}
    log(f"top8 classes: {top8}")

    sigma = sigma_for_epsilon(5.0, DELTA, T)
    log(f"T={T} sigma={sigma:.4f}")
    cfg = DistanceConfig(matching=MatchingConfig(unmatched_penalty=2.5, lambda_class=0.0),
                         gamma=0.5)

    rng = random.Random(0)
    rng_n = np.random.default_rng(3000)
    cands = seed_candidates(pool, K, MutationConfig(rng=random.Random(0)))
    rounds = []
    t0 = time.time()
    base = noisy_final = None
    for r in range(T):
        t_r = time.time()
        votes, cls_tv = score_round(priv, cands, cfg, idx, len(top8), tag=f"r{r}")
        noisy = np.clip(votes + rng_n.normal(0, sigma, len(votes)), 0, None)
        p = np.clip(votes, 0, None)
        p = p / p.sum()
        ess = 1.0 / float((p ** 2).sum())
        top10 = float(np.sort(p)[::-1][:10].sum())
        rec = {"round": r, "cands": len(cands), "nnz": int((votes > 0).sum()),
               "ESS": round(ess, 1), "top10": round(top10, 4),
               "per_record_class_TV": round(cls_tv, 4),
               "seconds": round(time.time() - t_r, 1)}
        rounds.append(rec)
        log(f"round {r:2d}: cands={len(cands)} nnz={rec['nnz']} ESS={rec['ESS']} "
            f"top10={rec['top10']} classTV={rec['per_record_class_TV']} "
            f"({rec['seconds']}s)")
        if r == T - 1:
            base, noisy_final = cands, noisy
            break
        cands = fv.evolve(cands, noisy, sel, pool, rng)

    w = fv.sampling_weights(noisy_final)
    dec = [base[j] for j in random.Random(500).choices(
        range(len(base)), weights=w.tolist(), k=M)]
    out = fv.s4(dec, noisy_final, base, 0.7, 3)
    norm = fv.metrics(priv, out, vocab)      # already PRIOR-normalised, NAMES keys
    avg = float(norm["Avg"])
    log(f"FINAL decoupled Avg={avg:.4f} :: {norm}")
    json.dump({"protocol": {"T": T, "eps": 5.0, "delta": DELTA, "sigma": sigma,
                            "lambda_class": 0.0, "release": "decoupled",
                            "seeds": 1, "noise_seed": 0, "K": K, "M": M},
               "rounds": rounds, "final": {"Avg": avg, "metrics": norm},
               "elapsed_s": round(time.time() - t0, 1)},
              open(OUT, "w"), indent=1)
    log(f"saved {OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
