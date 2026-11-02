# -*- coding: utf-8 -*-
"""P1: class-conditioned spatial relation fidelity  P(r | c_i, c_j).

Pre-declared rules (fixed before running)
  * class pairs are chosen by SUPPORT ONLY: all unordered pairs that occur in at
    least MIN_SUPPORT private object pairs, ranked by support, capped at MAX_PAIRS
  * relation set: the library's 7 relations (pair_relation)
  * per class pair and arm: TV(P_real(r|c_i,c_j), P_syn(r|c_i,c_j))
  * overall score = support-weighted mean TV (weights = real pair counts)
  * arms: Public-only, Chamfer+Card, LTSim+Card, CA-PE; same release protocol as
    the frozen table (5 DP seeds, 1110 layouts, S4 on the noisy support)
  * complexity: the same score restricted to private/released layouts with
    exactly n objects (n = 2..6)
"""
import io, os, sys, json, random, itertools, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
from collections import Counter, defaultdict
from layout.representation import Layout
from layout.relations import pair_relation, RELATIONS
from layout.mutation import MutationConfig, seed_candidates
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco
from evaluation.layout_metrics import class_distribution

MIN_SUPPORT, MAX_PAIRS = 100, 12
NAMES = ["Count", "Class", "Rel", "Ctr", "Size", "Scale"]
OUT = os.path.join(ROOT, "runs", "imgexp")
os.makedirs(OUT, exist_ok=True)
NSEEDS = list(range(5))
OUTNAME = "classcond_relation.json"
RIDX = {r: i for i, r in enumerate(RELATIONS)}


def cond_rel(layouts, pairs_filter=None):
    """P(rel | c_i, c_j) counts and pair support, optionally restricted."""
    acc = defaultdict(lambda: np.zeros(len(RELATIONS)))
    for L in layouts:
        objs = L.objects
        for a, b in itertools.combinations(objs, 2):
            key = tuple(sorted((a.cls, b.cls)))
            if pairs_filter is not None and key not in pairs_filter:
                continue
            acc[key][RIDX[pair_relation(a.box, b.box)]] += 1.0
    return {k: v for k, v in acc.items() if v.sum() > 0}


def main():
    global NSEEDS, OUTNAME
    if len(sys.argv) > 1 and sys.argv[1] == "--seeds":
        NSEEDS = [int(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 3 and sys.argv[3] == "--out":
        OUTNAME = sys.argv[4]
    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"),
                 encoding="utf-8")]
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    cands = seed_candidates(pool, 1000, MutationConfig(rng=random.Random(0)))
    K = len(cands)
    nobj = np.array([len(C) for C in cands], dtype=float)

    # ---- pre-declared class pairs (support only) ----
    ref = cond_rel(priv)
    support = {k: float(v.sum()) for k, v in ref.items()}
    ranked = sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = [k for k, s in ranked if s >= MIN_SUPPORT][:MAX_PAIRS]
    print(f"class pairs with support >= {MIN_SUPPORT}: "
          f"{sum(1 for _, s in ranked if s >= MIN_SUPPORT)}; using top {len(chosen)}")
    for k in chosen:
        print(f"   {k[0]} -- {k[1]:<14} support {int(support[k])}")
    pf = set(chosen)
    ref_f = {k: ref[k] / ref[k].sum() for k in chosen}

    # ---- arms ----
    votes = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))
    D_ch = np.load(os.path.join(ROOT, "runs", "dist_chamfer.npy"))
    D_lt = np.load(os.path.join(ROOT, "runs", "dist_ltsim_plain.npy"))

    def votes_from(D, mu, gamma):
        out = np.zeros(len(priv), dtype=np.int32)
        for i in range(len(priv)):
            m = len(priv[i].objects)
            d = D[i].astype(np.float64)
            if mu:
                d = d + mu * np.abs(m - nobj) / m
            if gamma:
                d = d + gamma * np.abs(m - nobj) / np.maximum(m, nobj)
            out[i] = int(np.argmin(d))
        return out

    arms = {
        "Public-only": np.ones(K),
        "Chamfer+Card": np.bincount(votes_from(D_ch, 2.5, 0.5), minlength=K).astype(float),
        "LTSim+Card": np.bincount(votes_from(D_lt, 2.5, 0.5), minlength=K).astype(float),
        "CA-PE": votes["ca"].astype(float),
    }

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    fv = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(fv)
    sigma = sigma_for_epsilon(5.0, 1e-5, 1)
    M = fv.M

    results = {}
    for arm, h in arms.items():
        t0 = time.time()
        per_seed, per_seed_rel = [], []
        for ns in NSEEDS:
            noisy = (h.astype(float) if arm == "Public-only"
                     else np.clip(h + np.random.default_rng(3000 + ns)
                                  .normal(0, sigma, K), 0, None))
            w = noisy / noisy.sum()
            rel_set = []
            for rs in range(12):
                rel = [cands[j] for j in random.Random(500 + rs).choices(
                    range(K), weights=w.tolist(), k=M)]
                rel_set.extend(fv.s4(rel, noisy, cands, 0.7, 3))
            got = cond_rel(rel_set, pf)
            tvs, ws = [], []
            for k in chosen:
                if k not in got:
                    tvs.append(1.0)
                    ws.append(support[k])
                    continue
                q = got[k] / got[k].sum()
                tvs.append(float(0.5 * np.abs(q - ref_f[k]).sum()))
                ws.append(support[k])
            per_seed.append(float(np.average(tvs, weights=ws)))
            per_seed_rel.append(dict(zip(["|".join(k) for k in chosen], tvs)))
        results[arm] = {"mean": float(np.mean(per_seed)),
                        "sd": float(np.std(per_seed, ddof=1)),
                        "per_seed": per_seed,
                        "per_pair": per_seed_rel[0]}
        print(f"{arm:14s} class-cond relation TV {np.mean(per_seed):.4f} "
              f"± {np.std(per_seed, ddof=1):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- complexity breakdown (CA-PE vs LTSim+Card, by object count) ----
    by_n = {}
    for n in range(2, 7):
        sub = [L for L in priv if len(L.objects) == n]
        if not sub:
            continue
        r_sub = cond_rel(sub, pf)
        r_f = {k: r_sub[k] / r_sub[k].sum() for k in chosen if k in r_sub}
        w_sub = {k: float(r_sub[k].sum()) for k in r_f}
        row = {"n_private": len(sub)}
        for arm in ("LTSim+Card", "CA-PE"):
            h = arms[arm]
            noisy = np.clip(h + np.random.default_rng(3000).normal(0, sigma, K), 0, None)
            w = noisy / noisy.sum()
            rel_set = []
            for rs in range(12):
                rel = [cands[j] for j in random.Random(500 + rs).choices(
                    range(K), weights=w.tolist(), k=M)]
                rel_set.extend(fv.s4(rel, noisy, cands, 0.7, 3))
            sub_rel = [L for L in rel_set if len(L.objects) == n]
            if not sub_rel or not r_f:
                continue
            g = cond_rel(sub_rel, set(r_f))
            tvs, ws = [], []
            for k in r_f:
                if k not in g:
                    tvs.append(1.0); ws.append(w_sub[k]); continue
                q = g[k] / g[k].sum()
                tvs.append(float(0.5 * np.abs(q - r_f[k]).sum()))
                ws.append(w_sub[k])
            row[arm] = float(np.average(tvs, weights=ws))
        by_n[n] = row
        if "CA-PE" in row and "LTSim+Card" in row:
            print(f"  n={n}: private {row['n_private']:4d}  "
                  f"LTSim+Card {row['LTSim+Card']:.3f}  CA-PE {row['CA-PE']:.3f}")

    json.dump({"classes_pairs": [list(k) for k in chosen],
               "support": {f"{k[0]}|{k[1]}": support[k] for k in chosen},
               "arms": results, "by_n": by_n,
               "noise_seeds": NSEEDS},
              open(os.path.join(OUT, OUTNAME), "w"), indent=1)
    print(f"\nsaved runs/imgexp/{OUTNAME}")


if __name__ == "__main__":
    main()
