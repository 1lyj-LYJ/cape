# -*- coding: utf-8 -*-
"""VG150 layout-completion (TSTR) under the same protocol as the COCO run.

Why: on COCO the public pool and the private set are both COCO, so Public-only
is an unusually strong reference (train2017 statistics vs val2017 layouts).
VG150 breaks that: the private layouts come from Visual Genome while the
candidate pool is still built from COCO train2017. If the DP gain grows with the
domain gap, that is a much stronger downstream claim.

Protocol (identical to scripts/lc_release.py + scripts/lc_task.py):
  split      80/20 of the 2409 VG layouts, split_seed 777 (same as COCO)
  synthesis  T=1, eps=5, delta=1e-5, sigma from the RDP accountant,
             1000 COCO-train2017 candidates (seed 0),
             5 outer noise seeds x 12 release resamples x M=1110,
             decoupled release + S4(t=0.7, seed 3)
  task       hide one object, predict centre and 4-way direction
  models     Ridge(alpha=1) + multinomial Logistic(C=1), 251 features
  sources    Simple prior, Public-only, Chamfer+Card, LTSim+Card, CA-PE, Real-train
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
import importlib.util

from layout.distance import DistanceConfig, layout_distance
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco

OUT = os.path.join(ROOT, "runs", "lc_vg")
LOG = os.path.join(OUT, "progress.log")
K, M, REPS, NSEED = 1000, 1110, 12, 5
DELTA = 1e-5
SPLIT_SEED = 777

_W = {}


def log(m):
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def _init(cands, ca_cfg):
    _W["c"] = cands
    _W["cfg"] = ca_cfg


def _chunk(privs):
    from layout.distance import DistanceConfig, layout_distance
    from layout.matching import MatchingConfig
    ch_cfg = DistanceConfig(matching=MatchingConfig(method="chamfer"), gamma=0.0)
    n, k = len(privs), len(_W["c"])
    A = np.zeros((n, k), dtype=np.float32)
    B = np.zeros((n, k), dtype=np.float32)
    for i, L in enumerate(privs):
        for j, Cm in enumerate(_W["c"]):
            A[i, j] = layout_distance(L, Cm, _W["cfg"])["total"]
            B[i, j] = layout_distance(L, Cm, ch_cfg)["total"]
    return A, B


def matrices(priv, cands, ca_cfg, jobs=6):
    p_ca = os.path.join(OUT, "dist_ca.npy")
    p_ch = os.path.join(OUT, "dist_chamfer.npy")
    if os.path.exists(p_ca) and os.path.exists(p_ch):
        log("  matrices: cached")
        return np.load(p_ca), np.load(p_ch)
    import multiprocessing as mp
    n_chunks = max(jobs, min(8 * jobs, len(priv)))
    size = (len(priv) + n_chunks - 1) // n_chunks
    chunks = [priv[i:i + size] for i in range(0, len(priv), size)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs, initializer=_init,
                  initargs=(list(cands), ca_cfg)) as pool:
        parts = pool.map(_chunk, chunks)
    A = np.vstack([p[0] for p in parts])
    B = np.vstack([p[1] for p in parts])
    np.save(p_ca, A)
    np.save(p_ch, B)
    log(f"  matrices: {A.shape} in {time.time()-t0:.0f}s")
    return A, B


def main():
    os.makedirs(OUT, exist_ok=True)
    open(LOG, "w", encoding="utf-8").close()
    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    fv = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(fv)
    spec2 = importlib.util.spec_from_file_location(
        "lct", os.path.join(ROOT, "scripts", "lc_task.py"))
    lct = importlib.util.module_from_spec(spec2)
    sys.argv = ["lct"]
    spec2.loader.exec_module(lct)

    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"),
                           encoding="utf-8"))
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "data/vg150_private/vg150_layouts_clean.jsonl"),
                 encoding="utf-8")]
    ids = [int(r["image_id"]) for r in rows]
    order = list(range(len(rows)))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(order)
    n_train = int(round(0.8 * len(order)))
    tr_idx = sorted(order[:n_train])
    te_idx = sorted(order[n_train:])
    priv = [Layout.from_dict(rows[i]) for i in tr_idx]
    test = [Layout.from_dict(rows[i]) for i in te_idx]
    log(f"VG split seed={SPLIT_SEED}: train {len(priv)} / test {len(test)}")

    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    cands = seed_candidates(pool, K, MutationConfig(rng=random.Random(0)))
    nobj = np.array([len(C) for C in cands], dtype=float)
    sigma = sigma_for_epsilon(5.0, DELTA, 1)
    log(f"sigma(eps=5,T=1)={sigma:.4f}")

    ca_cfg = DistanceConfig(matching=MatchingConfig(unmatched_penalty=2.5), gamma=0.5)
    D_ca, D_ch = matrices(priv, cands, ca_cfg)

    # LTSim: slice the cached full-VG matrix (row order == file order)
    p_lt = os.path.join(ROOT, "runs", "vgdist_ltsim_plain.npy")
    D_lt = np.load(p_lt)[tr_idx]
    log(f"LTSim matrix sliced: {D_lt.shape}")

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
        "CA-PE": np.bincount(votes_from(D_ca, 0.0, 0.0), minlength=K).astype(float),
    }
    for a, h in arms.items():
        log(f"  votes {a}: nnz={int((h > 0).sum())}")

    def dump(path, layouts, seed=None):
        with open(path, "w", encoding="utf-8") as f:
            for L in layouts:
                rec = {"objects": [{"cls": o.cls,
                                    "box": [o.box.x, o.box.y, o.box.w, o.box.h]}
                                   for o in L.objects]}
                if seed is not None:
                    rec["seed"] = seed
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dump(os.path.join(OUT, "real_train.jsonl"), priv)
    dump(os.path.join(OUT, "real_test.jsonl"), test)

    # real features (training references are fitted on the VG train split)
    Xr, Yr, Dr = lct.build([(None, [{"cls": o.cls, "box": [o.box.x, o.box.y, o.box.w, o.box.h]}
                                    for o in L.objects]) for L in priv])
    Xte, Yte, Dte = lct.build([(None, [{"cls": o.cls, "box": [o.box.x, o.box.y, o.box.w, o.box.h]}
                                       for o in L.objects]) for L in test])
    log(f"features: train {Xr.shape} test {Xte.shape}")

    res = {}
    # Simple prior fitted on the real train split
    prior_c = Yr.mean(axis=0, keepdims=True).repeat(len(Xte), axis=0)
    cerr = float(np.mean(np.linalg.norm(prior_c - Yte, axis=1)))
    vals, cnts = np.unique(Dr, return_counts=True)
    maj = int(vals[cnts.argmax()])
    acc = float((Dte == maj).mean())
    p_maj = float(cnts.max() / cnts.sum())
    pm = np.full((len(Dte), 4), (1 - p_maj) / 3)
    pm[:, maj] = p_maj
    ll = float(-np.mean(np.log(np.clip(pm[np.arange(len(Dte)), Dte], 1e-12, 1))))
    res["Simple prior"] = {"center_error": cerr, "dir_acc": acc, "dir_logloss": ll}
    log(f"Simple prior: {cerr:.4f} / {acc:.4f} / {ll:.4f}")

    # Real-train
    rt = {}
    lct.evaluate("Real-train", Xr, Yr, Dr, Xte, Yte, Dte, rt)
    res["Real-train"] = rt["Real-train"]
    log(f"Real-train: {res['Real-train']}")

    # synthetic arms
    for arm, h in arms.items():
        path = os.path.join(OUT, f"{arm.replace('+', '')}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for ns in range(NSEED):
                noisy = (h.astype(float) if arm == "Public-only" else
                         np.clip(h + np.random.default_rng(3000 + ns).normal(0, sigma, K), 0, None))
                w = noisy / noisy.sum()
                rel = []
                for rs in range(REPS):
                    rel += [cands[j] for j in random.Random(500 + rs).choices(
                        range(K), weights=w.tolist(), k=M)]
                out = fv.s4(rel, noisy, cands, 0.7, 3)
                for L in out:
                    f.write(json.dumps({"seed": ns, "objects": [
                        {"cls": o.cls, "box": [o.box.x, o.box.y, o.box.w, o.box.h]}
                        for o in L.objects]}, ensure_ascii=False) + "\n")
        log(f"  released {arm}")

        rows_syn = lct.load(path)
        per = []
        for s in range(NSEED):
            Xtr, Ytr, Dtr = lct.build(rows_syn, seeds={s})
            tmp = {}
            lct.evaluate(f"{arm}_s{s}", Xtr, Ytr, Dtr, Xte, Yte, Dte, tmp)
            per.append(tmp[f"{arm}_s{s}"])
        mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        sd = {k: float(np.std([p[k] for p in per], ddof=1)) for k in per[0]}
        res[arm] = {"mean": mean, "sd": sd, "per_seed": per}
        log(f"{arm}: acc={mean['dir_acc']:.4f}±{sd['dir_acc']:.4f} "
            f"centre={mean['center_error']:.4f} logloss={mean['dir_logloss']:.4f}")

    json.dump({"protocol": {"split_seed": SPLIT_SEED, "n_train": len(priv),
                            "n_test": len(test), "sigma": sigma, "eps": 5.0,
                            "delta": DELTA, "T": 1, "seeds": NSEED, "reps": REPS,
                            "release": M, "candidates": "COCO train2017"},
               "results": res},
              open(os.path.join(OUT, "completion_results_vg.json"), "w"), indent=1)
    log("ALL-DONE")


if __name__ == "__main__":
    main()
