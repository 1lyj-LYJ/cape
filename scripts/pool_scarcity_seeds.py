# -*- coding: utf-8 -*-
"""Pool-scarcity control, 3 outer noise seeds.

Same as scripts/pool_scarcity.py but repeats T=1 and T=12 at K=200 over three
noise seeds (rng 3000+ns), so the "evolution matters when the pool is scarce"
claim can be stated with a spread rather than a single draw.
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

from layout.distance import DistanceConfig, layout_distance
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco
from pe.selection import DiversitySelector, SelectionConfig, prior_strata_quotas
from evaluation.layout_metrics import layout_set_report

OUT = os.path.join(ROOT, "runs")
LOG = os.path.join(OUT, "pool_scarcity_seeds.log")
NAMES = ["Count", "Class", "Rel", "Ctr", "Size", "Scale"]
KEYS = ["count_TV", "class_TV", "relation_marginal_TV", "center_TV",
        "size_TV", "scale_W1"]
PRIOR = {"count_TV": 0.03253147953830014, "class_TV": 0.1426724971660939,
         "relation_marginal_TV": 0.14722313485315913,
         "center_TV": 0.12377635357751908, "size_TV": 0.15418084244481628,
         "scale_W1": 0.10755081712986059}
GEO = ["Rel", "Ctr", "Size", "Scale"]
M, REPS, K = 1110, 3, 200
DELTA = 1e-5
NSEED = 3

_W = {}


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def _init(cands, cfg):
    _W["c"] = cands
    _W["cfg"] = cfg


def _chunk(privs):
    from layout.distance import layout_distance
    out = np.zeros(len(_W["c"]))
    for L in privs:
        best_j, best_d = 0, float("inf")
        for j, Cm in enumerate(_W["c"]):
            d = layout_distance(L, Cm, _W["cfg"])["total"]
            if d < best_d:
                best_d, best_j = d, j
        out[best_j] += 1.0
    return out


def vote(priv, cands, cfg, jobs=6):
    import multiprocessing as mp
    n_chunks = max(jobs, min(8 * jobs, len(priv)))
    size = (len(priv) + n_chunks - 1) // n_chunks
    chunks = [priv[i:i + size] for i in range(0, len(priv), size)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_init, initargs=(list(cands), cfg)) as p:
        return sum(p.map(_chunk, chunks))


def run_arm(priv, cands0, T, ns, sel, pool, fv, vocab):
    sigma = sigma_for_epsilon(5.0, DELTA, T)
    rng = random.Random(0)
    rng_n = np.random.default_rng(3000 + ns)
    cands = list(cands0)
    for r in range(T):
        h = vote(priv, cands, cfg_global)
        noisy = np.clip(h + rng_n.normal(0, sigma, len(h)), 0, None)
        if r == T - 1:
            break
        cands = fv.evolve(cands, noisy, sel, pool, rng)
    w = fv.sampling_weights(noisy)
    vals = []
    for rs in range(REPS):
        rel = [cands[j] for j in random.Random(500 + rs).choices(
            range(len(cands)), weights=w.tolist(), k=M)]
        o = fv.s4(rel, noisy, cands, 0.7, 3)
        rep = layout_set_report(priv, o, vocab)
        vals.append({n: rep[k] / PRIOR[k] for n, k in zip(NAMES, KEYS)})
    m = {n: float(np.mean([v[n] for v in vals])) for n in NAMES}
    return {"T": T, "seed": ns, "Avg": float(np.mean(list(m.values()))),
            "Geo4": float(np.mean([m[n] for n in GEO])), "mean": m}


def main():
    open(LOG, "w", encoding="utf-8").close()
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    fv = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(fv)

    global cfg_global
    cfg_global = DistanceConfig(matching=MatchingConfig(unmatched_penalty=2.5), gamma=0.5)

    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"), encoding="utf-8")]
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    ff = {c: x / max(sum(pool.class_freq.values()), 1) for c, x in pool.class_freq.items()}
    strata = prior_strata_quotas(pool, freq=ff)
    sel = DiversitySelector(SelectionConfig(
        n_select=280, n_elite=280 // 3, lambda_redundancy=0.6,
        min_bucket_frac=0.12, rare_frac=0.30, strata_quotas=strata, freq=ff))
    cands0 = seed_candidates(pool, 1000, MutationConfig(rng=random.Random(0)))[:K]
    log(f"priv={len(priv)} K={K}")

    rows = []
    for T in (1, 12):
        for ns in range(NSEED):
            t0 = time.time()
            r = run_arm(priv, cands0, T, ns, sel, pool, fv, vocab)
            rows.append(r)
            log(f"K={K} T={T:2d} seed={ns}: Avg={r['Avg']:.4f} Geo4={r['Geo4']:.4f} "
                f"({time.time()-t0:.0f}s)")

    summary = {}
    for T in (1, 12):
        a = [r["Avg"] for r in rows if r["T"] == T]
        g = [r["Geo4"] for r in rows if r["T"] == T]
        summary[T] = {"Avg_mean": float(np.mean(a)), "Avg_sd": float(np.std(a, ddof=1)),
                      "Geo4_mean": float(np.mean(g)), "Geo4_sd": float(np.std(g, ddof=1))}
        log(f"T={T}: Avg={summary[T]['Avg_mean']:.4f}±{summary[T]['Avg_sd']:.4f} "
            f"Geo4={summary[T]['Geo4_mean']:.4f}±{summary[T]['Geo4_sd']:.4f}")

    json.dump({"K": K, "seeds": NSEED, "reps": REPS, "rows": rows, "summary": summary},
              open(os.path.join(OUT, "POOL_SCARCITY_3SEED.json"), "w"), indent=1)
    log("ALL-DONE")


if __name__ == "__main__":
    main()
