# -*- coding: utf-8 -*-
"""Regenerate DP layouts on the 80% private TRAIN split, for the TSTR task.

Frozen protocol, only the private subset changes (2287 of 2859 layouts):
  T=1, eps=5, delta=1e-5, sigma=1.0545, 1110 released layouts,
  decoupled release + S4 on the noisy support, 5 outer DP seeds.
Arms: Public-only, Chamfer+Card, LTSim+Card, CA-PE.
Also emits the real TRAIN layouts as the non-DP reference set.
"""
import io, os, sys, json, random, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import importlib.util
from layout.distance import DistanceConfig, layout_distance
from layout.ltsim import ltsim_distance
from layout.matching import MatchingConfig
from layout.mutation import MutationConfig, seed_candidates
from layout.representation import Layout
from pe.accountant import sigma_for_epsilon
from pe.public_pool import pool_from_coco

OUT = os.path.join(ROOT, "runs", "lc")
os.makedirs(OUT, exist_ok=True)
PROG = os.path.join(OUT, "progress.txt")
NSEED, REPS = 5, 12


def log(m):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def main():
    split = json.load(open(os.path.join(ROOT, "runs", "lc_split.json"),
                           encoding="utf-8"))
    train_ids = set(split["train_ids"])
    test_ids = set(split["test_ids"])
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"),
                 encoding="utf-8")]
    priv = [Layout.from_dict(r) for r in rows if int(r["image_id"]) in train_ids]
    test = [Layout.from_dict(r) for r in rows if int(r["image_id"]) in test_ids]
    log(f"train {len(priv)} layouts | test {len(test)} layouts")

    vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"),
                           encoding="utf-8"))
    pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                          vocab, max_sizes_per_class=30)
    cands = seed_candidates(pool, 1000, MutationConfig(rng=random.Random(0)))
    K = len(cands)
    nobj = np.array([len(C) for C in cands], dtype=float)

    spec = importlib.util.spec_from_file_location(
        "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
    fv = importlib.util.module_from_spec(spec)
    sys.argv = ["fv"]
    spec.loader.exec_module(fv)
    sigma = sigma_for_epsilon(5.0, 1e-5, 1)
    M = fv.M

    def matrix(tag, fn):
        p = os.path.join(OUT, f"dist_{tag}.npy")
        if os.path.exists(p):
            log(f"  matrix {tag}: cached")
            return np.load(p)
        t0 = time.time()
        D = np.zeros((len(priv), K), dtype=np.float32)
        for i, L in enumerate(priv):
            for j, C in enumerate(cands):
                D[i, j] = fn(L, C)
            if (i + 1) % 400 == 0:
                log(f"  matrix {tag}: {i+1}/{len(priv)} ({time.time()-t0:.0f}s)")
        np.save(p, D)
        log(f"  matrix {tag}: done {time.time()-t0:.0f}s")
        return D

    ca_cfg = DistanceConfig(matching=MatchingConfig(unmatched_penalty=2.5), gamma=0.5)
    D_ca = matrix("ca", lambda L, C: layout_distance(L, C, ca_cfg)["total"])
    D_ch = matrix("chamfer", lambda L, C: layout_distance(
        L, C, DistanceConfig(matching=MatchingConfig(method="chamfer"),
                             gamma=0.0))["total"])
    D_lt = matrix("ltsim", lambda L, C: ltsim_distance(L, C))

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

    def dump(path, layouts):
        with open(path, "w", encoding="utf-8") as f:
            for L in layouts:
                f.write(json.dumps({"objects": [
                    {"cls": o.cls, "box": [o.box.x, o.box.y, o.box.w, o.box.h]}
                    for o in L.objects]}, ensure_ascii=False) + "\n")

    dump(os.path.join(OUT, "real_train.jsonl"), priv)
    dump(os.path.join(OUT, "real_test.jsonl"), test)

    for arm, h in arms.items():
        per_seed_unique = []
        with open(os.path.join(OUT, f"{arm.replace('+','').replace('-','')}.jsonl"),
                  "w", encoding="utf-8") as f:
            for ns in range(NSEED):
                noisy = (h.astype(float) if arm == "Public-only"
                         else np.clip(h + np.random.default_rng(3000 + ns)
                                      .normal(0, sigma, K), 0, None))
                w = noisy / noisy.sum()
                rel = []
                for rs in range(REPS):
                    rel += [cands[j] for j in random.Random(500 + rs).choices(
                        range(K), weights=w.tolist(), k=M)]
                out = fv.s4(rel, noisy, cands, 0.7, 3)
                sig = {tuple((o.cls, round(o.box.x, 4), round(o.box.y, 4))
                             for o in L.objects) for L in out}
                per_seed_unique.append(len(sig))
                for L in out:
                    f.write(json.dumps({"seed": ns, "objects": [
                        {"cls": o.cls,
                         "box": [o.box.x, o.box.y, o.box.w, o.box.h]}
                        for o in L.objects]}, ensure_ascii=False) + "\n")
        log(f"{arm:14s} released {NSEED} x {M} layouts, unique/seed "
            f"{per_seed_unique}")

    json.dump({"n_train": len(priv), "n_test": len(test), "sigma": sigma,
               "seeds": NSEED, "reps": REPS, "release": M},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    log("ALL-DONE")


if __name__ == "__main__":
    main()
