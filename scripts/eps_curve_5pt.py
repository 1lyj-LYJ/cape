# -*- coding: utf-8 -*-
"""Full privacy-budget curve for CA-PE at T=1: eps = 1,2,3,4,5, n=10 seeds.

Same protocol as the main table (decoupled release, 12 inner resamples, S4
t=0.7 seed 3); only sigma changes with epsilon. Also emits a plot-ready CSV and
a PNG so the manuscript figure can be swapped in directly.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
REPS, SEEDS = 12, list(range(10))
EPS = [1.0, 2.0, 3.0, 4.0, 5.0]

vocab = json.load(open(os.path.join(ROOT, "configs/coco80_classes.json"), encoding="utf-8"))
priv = [Layout.from_dict(json.loads(l)) for l in
        open(os.path.join(ROOT, "data/coco_private/coco_layouts.jsonl"), encoding="utf-8")]
pool = pool_from_coco(os.path.join(ROOT, "data/raw/annotations/instances_train2017.json"),
                      vocab, max_sizes_per_class=30)
cands = seed_candidates(pool, 1000, MutationConfig(rng=random.Random(0)))
K = len(cands)

spec = importlib.util.spec_from_file_location(
    "fv", os.path.join(ROOT, "scripts", "final_validation.py"))
fv = importlib.util.module_from_spec(spec)
sys.argv = ["fv"]
spec.loader.exec_module(fv)
M = fv.M

h = np.load(os.path.join(ROOT, "runs", "real_votes.npz"))["ca"].astype(float)

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
    sd = {n: float(np.std([s[n] for s in per], ddof=1)) for n in NAMES}
    out[f"{eps:g}"] = {"sigma": sigma, "mean": mean, "seed_sd": sd,
                       "Avg": float(np.mean([mean[n] for n in NAMES])),
                       "Avg_sd": float(np.std([np.mean([s[n] for n in NAMES]) for s in per], ddof=1)),
                       "Geo4": float(np.mean([mean[n] for n in GEO])),
                       "Geo4_sd": float(np.std([np.mean([s[n] for n in GEO]) for s in per], ddof=1)),
                       "per_seed": per}
    print(f"eps={eps:g}  sigma={sigma:.4f}  Avg={out[f'{eps:g}']['Avg']:.4f}  "
          f"Geo4={out[f'{eps:g}']['Geo4']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

json.dump({"protocol": {"T": 1, "delta": 1e-5, "seeds": SEEDS,
                        "inner_resamples": REPS, "release": "decoupled",
                        "s4": {"t": 0.7, "seed": 3}}, "eps": EPS, "rows": out},
          open(os.path.join(ROOT, "runs", "EPS_CURVE_5PT.json"), "w"), indent=1)

# ---- plot-ready CSV
with open(os.path.join(ROOT, "runs", "EPS_CURVE_5PT.csv"), "w", encoding="utf-8") as f:
    f.write("epsilon,sigma,Avg,Avg_sd,Geo4,Geo4_sd\n")
    for eps in EPS:
        d = out[f"{eps:g}"]
        f.write(f"{eps:g},{d['sigma']:.4f},{d['Avg']:.4f},{d['Avg_sd']:.4f},"
                f"{d['Geo4']:.4f},{d['Geo4_sd']:.4f}\n")

# ---- figure in the style of the manuscript panel
xs = EPS
avg = [out[f"{e:g}"]["Avg"] for e in xs]
avg_sd = [out[f"{e:g}"]["Avg_sd"] for e in xs]
geo = [out[f"{e:g}"]["Geo4"] for e in xs]
geo_sd = [out[f"{e:g}"]["Geo4_sd"] for e in xs]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
ax.errorbar(xs, avg, yerr=avg_sd, color="#1f4e79", marker="o", ms=6, lw=1.8,
            capsize=3, label="Avg")
ax.errorbar(xs, geo, yerr=geo_sd, color="#c55a11", marker="s", ms=5, lw=1.6,
            ls="--", capsize=3, label="Geo4")
for x, y in zip(xs, avg):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, color="#1f4e79")
for x, y in zip(xs, geo):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=9, color="#c55a11")
ax.set_xlabel("Privacy budget ε")
ax.set_ylabel("Normalized error (lower is better)")
ax.set_xticks(xs)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_ylim(0.44, 0.78)
ax.grid(True, which="major", axis="y", color="#dddddd", lw=0.7)
ax.legend(frameon=False, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
png = os.path.join(ROOT, "paper", "figs", "fig5_eps_curve_5pt.png")
os.makedirs(os.path.dirname(png), exist_ok=True)
fig.savefig(png)
print("saved:", png)
print("saved: runs/EPS_CURVE_5PT.json / .csv")
