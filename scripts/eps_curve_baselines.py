# -*- coding: utf-8 -*-
"""Privacy-utility curve for the baselines: one method, eps = 1..5, n=10 seeds.

The vote histogram of a method does not depend on epsilon, so only sigma and the
release loop change. Vote histograms are rebuilt exactly as in
scripts/unified_main_table.py so that the eps=5 row reproduces the main table.

Usage:  py eps_curve_baselines.py {ca|lt|cch}
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
REPS, SEEDS, EPS = 12, list(range(10)), [1.0, 2.0, 3.0, 4.0, 5.0]

ARG = sys.argv[1] if len(sys.argv) > 1 else "lt"
LABEL = {"ca": "CA-PE", "lt": "LTSim+Card", "cch": "Chamfer+Card"}[ARG]

vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
priv = [Layout.from_dict(json.loads(l)) for l in
        open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"), encoding="utf-8")]
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
M = fv.M


def votes_from_matrix(D, mu=2.5, gamma=0.5):
    out = np.zeros(len(priv), dtype=np.int32)
    for i in range(len(priv)):
        m = len(priv[i].objects)
        d = D[i].astype(np.float64)
        d = d + mu * np.abs(m - nobj) / m
        d = d + gamma * np.abs(m - nobj) / np.maximum(m, nobj)
        out[i] = int(np.argmin(d))
    return out


if ARG == "ca":
    h = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))["ca"].astype(float)
elif ARG == "lt":
    D = np.load(os.path.join(ROOT, "runs", "dist_ltsim_plain.npy"))
    h = np.bincount(votes_from_matrix(D), minlength=K).astype(float)
else:
    D = np.load(os.path.join(ROOT, "runs", "dist_chamfer.npy"))
    h = np.bincount(votes_from_matrix(D), minlength=K).astype(float)
print(f"[{ARG}] histogram nnz={int((h > 0).sum())}", flush=True)

out = {}
for eps in EPS:
    sigma = sigma_for_epsilon(eps, 1e-5, 1)
    t0 = time.time()
    per = []
    for ns in SEEDS:
        noisy = np.clip(h + np.random.default_rng(3000 + ns).normal(0, sigma, K), 0, None)
        w = noisy / noisy.sum()
        vals = []
        for rs in range(REPS):
            rel = [cands[j] for j in random.Random(500 + rs).choices(
                range(K), weights=w.tolist(), k=M)]
            o = fv.s4(rel, noisy, cands, 0.7, 3)
            rep = layout_set_report(priv, o, vocab)
            vals.append({n: rep[k] / PRIOR[k] for n, k in zip(NAMES, KEYS)})
        per.append({n: float(np.mean([v[n] for v in vals])) for n in NAMES})
    mean = {n: float(np.mean([s[n] for s in per])) for n in NAMES}
    out[f"{eps:g}"] = {"sigma": sigma, "mean": mean,
                       "Avg": float(np.mean([mean[n] for n in NAMES])),
                       "Avg_sd": float(np.std([np.mean([s[k] for k in NAMES]) for s in per],
                                              ddof=1)),
                       "Geo4": float(np.mean([mean[n] for n in GEO])),
                       "Geo4_sd": float(np.std([np.mean([s[k] for k in GEO]) for s in per],
                                               ddof=1)),
                       "per_seed": per}
    print(f"[{ARG}] eps={eps:g} sigma={sigma:.4f} Avg={out[f'{eps:g}']['Avg']:.4f} "
          f"Geo4={out[f'{eps:g}']['Geo4']:.4f} ({time.time()-t0:.0f}s)", flush=True)

json.dump({"method": ARG, "label": LABEL, "eps": EPS, "rows": out},
          open(os.path.join(ROOT, "runs", f"EPS_CURVE_{ARG}.json"), "w"), indent=1)
print(f"saved runs/EPS_CURVE_{ARG}.json")
