# -*- coding: utf-8 -*-
"""Noisy (full DP) replication of the matching-ablation variants.

Question: do the component effects measured on the *support layer* (noise-free,
deterministic, runs/ABLATION_MATCHING.md) survive the actual DP pipeline
(noise -> support-weighted release -> S4)?

Protocol (fixed before running):
  * same eight variants as the support-layer ablation, same cost settings
  * T=1, eps=5, delta=1e-5, sigma=1.0545 (RDP accountant)
  * K=1000 candidates (seed 0), release M=1110, S4 t=0.7 seed 3
  * 10 outer noise seeds (rng 3000+ns), 12 inner release resamples each
  * primary endpoint: Avg; paired comparison against `full` on the same seeds
  * all six metrics reported; no variant dropped, no seed dropped

Integrity anchor: `full` must reproduce the CA-PE row of
runs/UNIFIED_TABLE_10SEED.json (0.6360) bit for bit.

Usage:  py ablation_noisy.py <variant>
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

from layout.distance import DistanceConfig
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.parallel_vote import vote_histogram
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
REPS, NSEED, M = 12, 10, 1110

VARIANTS = {
    "full":      dict(mu=2.5, gamma=0.5, lc=1.0, lctr=2.0, lsz=1.0, lgiou=1.0, method="hungarian"),
    "no_card":   dict(mu=0.0, gamma=0.0, lc=1.0, lctr=2.0, lsz=1.0, lgiou=1.0, method="hungarian"),
    "soft_card": dict(mu=1.0, gamma=0.2, lc=1.0, lctr=2.0, lsz=1.0, lgiou=1.0, method="hungarian"),
    "no_class":  dict(mu=2.5, gamma=0.5, lc=0.0, lctr=2.0, lsz=1.0, lgiou=1.0, method="hungarian"),
    "no_center": dict(mu=2.5, gamma=0.5, lc=1.0, lctr=0.0, lsz=1.0, lgiou=1.0, method="hungarian"),
    "no_size":   dict(mu=2.5, gamma=0.5, lc=1.0, lctr=2.0, lsz=0.0, lgiou=1.0, method="hungarian"),
    "no_giou":   dict(mu=2.5, gamma=0.5, lc=1.0, lctr=2.0, lsz=1.0, lgiou=0.0, method="hungarian"),
    "rand_match": dict(mu=2.5, gamma=0.5, lc=1.0, lctr=2.0, lsz=1.0, lgiou=1.0, method="random"),
}


def log(msg):
    print(msg, flush=True)


def main():
    name = _ARGV[1]
    v = VARIANTS[name]
    t0 = time.time()
    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"), encoding="utf-8")]
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    cands = seed_candidates(pool, 1000, MutationConfig(rng=random.Random(0)))
    K = len(cands)

    vp = os.path.join(ROOT, "runs", f"abl_noisy_votes_{name}.npz")
    if os.path.exists(vp):
        h = np.load(vp)["h"].astype(float)
        log(f"[{name}] votes cached nnz={int((h > 0).sum())}")
    else:
        cfg = DistanceConfig(
            matching=MatchingConfig(lambda_class=v["lc"], lambda_center=v["lctr"],
                                    lambda_size=v["lsz"], lambda_giou=v["lgiou"],
                                    unmatched_penalty=v["mu"], method=v["method"]),
            gamma=v["gamma"])
        h, _ = vote_histogram(priv, cands, cfg, n_jobs=1)
        np.savez(vp, h=h)
        log(f"[{name}] votes computed nnz={int((h > 0).sum())} ({time.time()-t0:.0f}s)")

    if name == "full":
        cached = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))["ca"].astype(float)
        if not np.array_equal(h, cached):
            raise SystemExit("INTEGRITY FAIL: full does not reproduce cached votes")
        log("[full] votes == cached real_votes['ca'] (bit-identical)")

    sigma = sigma_for_epsilon(5.0, 1e-5, 1)
    per = []
    for ns in range(NSEED):
        noisy = np.clip(h + np.random.default_rng(3000 + ns).normal(0, sigma, K), 0, None)
        w = noisy / noisy.sum()
        vals = []
        for rs in range(REPS):
            rel = [cands[j] for j in random.Random(500 + rs).choices(
                range(K), weights=w.tolist(), k=M)]
            o = fv_s4(rel, noisy, cands)
            rep = layout_set_report(priv, o, vocab)
            vals.append({n: rep[k] / PRIOR[k] for n, k in zip(NAMES, KEYS)})
        per.append({n: float(np.mean([x[n] for x in vals])) for n in NAMES})
        log(f"[{name}] seed {ns} done ({time.time()-t0:.0f}s)")

    mean = {n: float(np.mean([s[n] for s in per])) for n in NAMES}
    sd = {n: float(np.std([s[n] for s in per], ddof=1)) for n in NAMES}
    out = {"variant": name, "config": v, "sigma": sigma,
           "seeds": NSEED, "reps": REPS, "votes_nnz": int((h > 0).sum()),
           "mean": mean, "seed_sd": sd,
           "Avg": float(np.mean([mean[n] for n in NAMES])),
           "Geo4": float(np.mean([mean[n] for n in GEO])),
           "per_seed": per}
    json.dump(out, open(os.path.join(ROOT, "runs", f"abl_noisy_{name}.json"), "w"), indent=1)
    log(f"[{name}] Avg={out['Avg']:.4f} Geo4={out['Geo4']:.4f} total {time.time()-t0:.0f}s")


import importlib.util

_ARGV = list(sys.argv)          # saved before the frozen module resets sys.argv
_spec = importlib.util.spec_from_file_location(
    "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
_fv = importlib.util.module_from_spec(_spec)
sys.argv = ["fv"]
_spec.loader.exec_module(_fv)


def fv_s4(rels, support, cands):
    """The frozen S4, so every variant uses the same code path."""
    return _fv.s4(rels, support, cands, 0.7, 3)


if __name__ == "__main__":
    main()
