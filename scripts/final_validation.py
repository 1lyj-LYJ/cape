# -*- coding: utf-8 -*-
"""FINAL VALIDATION — decoupled exploration/release vs the original release,
for both backbones, at T=1 and T=12, both with the same S4.

Design (one engine pass yields BOTH arms, so they share the same DP support):
  round 0..T-2 : vote -> evolve              (unchanged exploration)
  round T-1    : vote on C_{T-1}  -> support S
                 original  = evolve once more  -> C_T (parents+children+fresh)
                 decoupled = resample C_{T-1} by S  (no truncation/injection)
                 both then get the SAME S4 class correction driven by S
Exactly T DP queries either way.

Usage:  py final_validation.py 1 ca|lt     (T=1 uses cached real votes)
        py final_validation.py 12 ca|lt    (full loop, slow)
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
from collections import Counter

from evaluation.layout_metrics import layout_set_report
from layout.distance import DistanceConfig
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, mutate_n, seed_candidates
from layout.representation import Box, Layout, Object
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco
from pe.selection import (DiversitySelector, SelectionConfig, prior_strata_quotas)


PRIOR_COUNT = 0.03253147953830014


def _cd(layouts, max_count=8):
    """Mirror of evaluation.layout_metrics.count_distribution."""
    c = Counter(min(len(L), max_count) for L in layouts)
    v = np.array([c.get(k, 0) for k in range(1, max_count + 1)], dtype=float)
    return v / v.sum() if v.sum() > 0 else v


def _sig(L):
    return (tuple(sorted(o.cls for o in L.objects)),
            tuple(round(o.box.area, 6)
                  for o in sorted(L.objects, key=lambda o: (o.box.cx, o.box.cy))))


def _diag_block(support, base, out, priv):
    """Case A (support already skewed) vs Case B (sampling variance)?"""
    p_ = np.clip(np.asarray(support, dtype=float), 0, None)
    p_ = p_ / p_.sum()
    ess = float(1.0 / np.sum(p_ ** 2))
    r_priv = _cd(priv)
    q = np.zeros(8)
    for j, L in enumerate(base):
        q[min(len(L), 8) - 1] += p_[j]
    raw_th = float(np.abs(q - r_priv).sum() / 2.0)
    raw_act = float(np.abs(_cd(out) - r_priv).sum() / 2.0)
    sg = Counter(_sig(L) for L in out)
    return {
        "support_size": int(len(p_)),
        "support_nonzero": int((p_ > 1e-9).sum()),
        "ESS": round(ess, 1),
        "top1_share": round(float(p_.max()), 4),
        "count_metric_theoretical": round(raw_th / PRIOR_COUNT, 3),
        "count_metric_actual": round(raw_act / PRIOR_COUNT, 3),
        "count_raw_tv_theoretical": round(raw_th, 5),
        "count_raw_tv_actual": round(raw_act, 5),
        "release_size": len(out),
        "unique_layouts": len(sg),
        "max_multiplicity": int(max(sg.values())) if sg else 0,
        "count_hist_private": [round(x, 4) for x in r_priv.tolist()],
        "count_hist_theoretical": [round(x, 4) for x in q.tolist()],
        "count_hist_actual": [round(x, 4) for x in _cd(out).tolist()],
    }

KEYS = ["count_TV", "class_TV", "relation_marginal_TV", "center_TV",
        "size_TV", "scale_W1"]
NAMES = ["Count", "Class", "Rel", "Ctr", "Size", "Scale"]
PRIOR = {"count_TV": 0.03253147953830014, "class_TV": 0.1426724971660939,
         "relation_marginal_TV": 0.14722313485315913,
         "center_TV": 0.12377635357751908, "size_TV": 0.15418084244481628,
         "scale_W1": 0.10755081712986059}
PROG = os.path.join(ROOT, 'runs', 'final_progress.txt')


def _prog(msg):
    with open(PROG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg, flush=True)


K, PARENTS, FRESH, M = 1000, 280, 550, 1110
DELTA = 1e-5


def sampling_weights(support):
    """Non-negative clip, then normalize; uniform fallback if all mass is zero.

    The fallback is fixed a priori (uniform over the public candidate set) and
    depends only on public information, so it stays post-processing. It is not
    triggered anywhere in the reported experiments; it exists so that every
    possible noise realisation yields a defined sampling distribution.
    """
    v = np.clip(np.asarray(support, dtype=float), 0.0, None)
    tot = float(v.sum())
    if tot <= 0.0:
        return np.full(len(v), 1.0 / max(len(v), 1))
    return v / tot


def s4(rels, support, cands, t=0.7, seed=3):
    rng = random.Random(seed)
    w = sampling_weights(support)
    vh = Counter()
    for j, wj in enumerate(w):
        for o in cands[j].objects:
            vh[o.cls] += wj
    vs = sum(vh.values()) or 1.0
    cf = Counter()
    for L in cands:
        for o in L.objects:
            cf[o.cls] += 1
    tot = sum(cf.values()) or 1.0
    # deterministic across processes: iterate classes in sorted order.
    # (set iteration follows PYTHONHASHSEED and used to leak into the bag
    # order, padding tie-breaks and truncation tail -> Class drifted ~4e-3)
    tgt = {c: t * cf[c] / tot + (1 - t) * vh.get(c, 0.0) / vs
           for c in sorted(set(vh) | set(cf))}
    ts = sum(tgt.values()) or 1.0
    tgt = {c: v / ts for c, v in tgt.items()}
    slots = sum(len(L.objects) for L in rels)
    bag = []
    for c in sorted(tgt):
        bag += [c] * int(round(tgt[c] * slots))
    fill = max(sorted(tgt), key=tgt.get)   # sorted => deterministic tie-break
    while len(bag) < slots:
        bag.append(fill)
    bag = bag[:slots]
    rng.shuffle(bag)
    out, p = [], 0
    for L in rels:
        objs = [Object(cls=bag[p + i], box=Box(o.box.x, o.box.y, o.box.w, o.box.h))
                for i, o in enumerate(L.objects)]
        p += len(L.objects)
        out.append(Layout(objects=objs))
    return out


def metrics(priv, rels, vocab):
    rep = layout_set_report(priv, rels, vocab)
    n = [float(rep[k]) / PRIOR[k] for k in KEYS]
    return {"Avg": sum(n) / 6, **{a: b for a, b in zip(NAMES, n)}}


def evolve(cands, noisy, sel, pool, rng, n_children=PARENTS, n_fresh=FRESH):
    pidx = sel(noisy, cands, rng)
    v = np.clip(noisy, 0, None); vmax = float(v.max()) if len(v) else 1.0
    fit = [float(v[i]) / vmax for i in pidx]
    thr = float(np.quantile(fit, 0.25)) if fit else 0.0
    cfg = MutationConfig(rng=rng)
    kids = []
    for k in range(n_children):
        i = pidx[k % len(pidx)]
        if fit[k % len(fit)] >= thr:
            kids.append(mutate_n(cands[i], pool, 1, cfg, None, mode="refine"))
        else:
            kids.append(mutate_n(cands[i], pool, rng.randint(1, 3), cfg, None, mode="full"))
    fresh = seed_candidates(pool, n_fresh, cfg) if n_fresh else []
    return [cands[i] for i in pidx] + kids + fresh


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    bb = sys.argv[2] if len(sys.argv) > 2 else "ca"
    NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else (5 if T == 1 else 3)
    NOISE_SEEDS = ([int(x) for x in sys.argv[4].split(",")]
                   if len(sys.argv) > 4 else list(range(NSEED)))
    NS_TAG = ("_ns" + "_".join(str(x) for x in NOISE_SEEDS)
              if len(sys.argv) > 4 else "")
    sigma = sigma_for_epsilon(5.0, DELTA, T)
    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
    priv = [Layout.from_dict(json.loads(l)) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"),
                 encoding="utf-8")][:2859]
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    ff = {c: x / max(sum(pool.class_freq.values()), 1) for c, x in pool.class_freq.items()}
    strata = prior_strata_quotas(pool, freq=ff)
    _prog(f"T={T} backbone={bb} sigma={sigma:.4f}")

    def make_sel(seed):
        return DiversitySelector(SelectionConfig(
            n_select=PARENTS, n_elite=PARENTS // 3, lambda_redundancy=0.6,
            min_bucket_frac=0.12, rare_frac=0.30, strata_quotas=strata, freq=ff))

    if T == 1:
        votes = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))["ltsim" if bb == "lt" else "ca"]
        rows = {"original": [], "decoupled": []}
        for ns in range(5):
            noisy = np.clip(votes + np.random.default_rng(3000 + ns).normal(0, sigma, len(votes)),
                            0, None)
            w = sampling_weights(noisy)
            for rs in range(3):
                rng = random.Random(rs)
                cands = seed_candidates(pool, K, MutationConfig(rng=random.Random(0)))
                orig = evolve(cands, noisy, make_sel(rs), pool, rng)
                dec = [cands[j] for j in random.Random(500 + rs).choices(
                    range(K), weights=w.tolist(), k=M)]
                for tag, rel in (("original", orig), ("decoupled", dec)):
                    out = s4(rel, noisy, cands, 0.7, 3)
                    rows[tag].append({**metrics(priv, out, vocab), "noise": ns})
            _prog(f"T1 noise {ns} done")
    else:
        votes = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))["ltsim" if bb == "lt" else "ca"]   # round-0 votes
        rows = {"original": [], "decoupled": []}
        for ns in NOISE_SEEDS:
            rng_n = np.random.default_rng(3000 + ns)
            for rs in range(1):
                t0 = time.time()
                cands = seed_candidates(pool, K, MutationConfig(rng=random.Random(0)))
                rng = random.Random(rs)
                sel = make_sel(rs)
                noisy_final = None
                for r in range(T):
                    if r == 0 and bb == "ca":
                        true_v = votes
                    else:
                        from pe.parallel_vote import vote_histogram
                        if bb == "ca":
                            true_v, _ = vote_histogram(
                                priv, cands,
                                DistanceConfig(matching=MatchingConfig(unmatched_penalty=2.5),
                                               gamma=0.5), n_jobs=1)
                        else:
                            from layout.ltsim import ltsim_distance
                            true_v = np.zeros(len(cands))
                            for L in priv:
                                m = max(len(L), 1)
                                bj, bd = 0, float("inf")
                                for j, c in enumerate(cands):
                                    d = ltsim_distance(L, c) + 2.5 * abs(len(L) - len(c)) / m
                                    if d < bd:
                                        bd, bj = d, j
                                true_v[bj] += 1
                    noisy = np.clip(true_v + rng_n.normal(0, sigma, len(true_v)), 0, None)
                    if r == T - 1:
                        noisy_final = noisy
                        base = cands
                        break
                    cands = evolve(cands, noisy, sel, pool, rng)
                    _prog(f"  seed {ns} round {r} done ({time.time()-t0:.0f}s)")
                w = sampling_weights(noisy_final)
                _diag = {} if os.environ.get("DIAG") else None
                orig = evolve(base, noisy_final, sel, pool, rng)
                dec = [base[j] for j in random.Random(500 + rs).choices(
                    range(len(base)), weights=w.tolist(), k=M)]
                for tag, rel in (("original", orig), ("decoupled", dec)):
                    out = s4(rel, noisy_final, base, 0.7, 3)
                    rows[tag].append({**metrics(priv, out, vocab), "noise": ns})
                    if _diag is not None:
                        try:
                            _diag[tag] = _diag_block(noisy_final, base, out, priv)
                        except Exception as e:
                            _diag[tag] = {"error": repr(e)}
                if _diag is not None:
                    try:
                        json.dump(_diag, open(os.path.join(
                            ROOT, "runs", f"diag_T{T}_{bb}_ns{ns}.json"), "w"), indent=1)
                    except Exception:
                        pass
            _prog(f"seed {ns} DONE")

    print(f"\n{'arm':<12}{'Avg':>16}{'Count':>8}{'Class':>8}{'Rel':>8}"
          f"{'Ctr':>8}{'Size':>8}{'Scale':>8}")
    res = {}
    for tag, rs_ in rows.items():
        by = {}
        for r in rs_:
            by.setdefault(r["noise"], []).append(r["Avg"])
        per = [float(np.mean(v)) for _, v in sorted(by.items())]
        m = {n: float(np.mean([r[n] for r in rs_])) for n in NAMES}
        print(f"{tag:<12}{np.mean(per):>8.3f}±{np.std(per):.3f}"
              + "".join(f"{m[n]:>8.3f}" for n in NAMES))
        res[tag] = {"Avg": round(float(np.mean(per)), 4),
                    "Avg_std": round(float(np.std(per)), 4),
                    **{n: round(m[n], 4) for n in NAMES},
                    "per_noise": [round(x, 4) for x in per]}
    o = np.array(res["original"]["per_noise"]); d = np.array(res["decoupled"]["per_noise"])
    if len(o) == len(d):
        from scipy import stats as st
        diff = o - d
        t = st.ttest_rel(o, d)
        print(f"\nDelta_release = {diff.mean():+.4f}±{diff.std():.4f}  "
              f"t={t.statistic:+.2f} p={t.pvalue:.4f}  decoupled better in "
              f"{int((diff>0).sum())}/{len(diff)}")
    json.dump(res, open(os.path.join(ROOT, "runs", f"final_T{T}_{bb}{NS_TAG}.json"), "w"), indent=1)
    _prog("RESULTS " + json.dumps(res))


if __name__ == "__main__":
    main()
