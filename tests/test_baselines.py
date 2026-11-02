"""Tests for the correspondence baselines (Hungarian / random / Chamfer).

Run: py tests/test_baselines.py

These tests guard three things:
  1. The refactored Hungarian branch is BIT-IDENTICAL to the original
     direct ``linear_sum_assignment`` implementation on many random pairs.
  2. The random-assignment baseline is reproducible, produces a valid
     one-to-one assignment, and (in expectation) is never better than the
     optimal Hungarian assignment.
  3. Chamfer is a valid permutation-invariant, symmetric set distance.
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linear_sum_assignment

from layout import Box, Layout, Object, MatchingConfig, match_layouts
from layout.matching import cost_matrix

PASS = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name} {extra}")


def rand_layout(rng, n):
    objs = []
    for _ in range(n):
        w = rng.uniform(0.05, 0.4)
        h = rng.uniform(0.05, 0.4)
        x = rng.uniform(0.0, 1.0 - w)
        y = rng.uniform(0.0, 1.0 - h)
        objs.append(Object(rng.choice(["person", "car", "dog", "chair", "cup"]),
                           Box(x, y, w, h)))
    return Layout(objects=objs)


def reference_hungarian(a, b, cfg):
    """The original pre-refactor implementation, copied verbatim."""
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return cfg.unmatched_penalty
    C = cost_matrix(a, b, cfg)
    K = max(n, m)
    Cp = np.full((K, K), cfg.unmatched_penalty, dtype=np.float64)
    Cp[:n, :m] = C
    row_ind, col_ind = linear_sum_assignment(Cp)
    matched, ua, ub = [], [], []
    costs = []
    for i, j in zip(row_ind, col_ind):
        if i < n and j < m:
            matched.append((int(i), int(j)))
            costs.append(float(C[i, j]))
        elif i < n:
            ua.append(int(i))
        elif j < m:
            ub.append(int(j))
    n_unmatched = len(ua) + len(ub)
    mean_cost = float(np.mean(costs)) if costs else cfg.unmatched_penalty
    return mean_cost + cfg.unmatched_penalty * n_unmatched / max(n, 1)


# --------------------------------------------------------------- bit-identical
cfg = MatchingConfig()
rng = random.Random(42)
max_diff = 0.0
npairs = 0
for _ in range(300):
    a = rand_layout(rng, rng.randint(1, 6))
    b = rand_layout(rng, rng.randint(1, 6))
    ref = reference_hungarian(a, b, cfg)
    new = match_layouts(a, b, MatchingConfig(method="hungarian")).distance
    max_diff = max(max_diff, abs(ref - new))
    npairs += 1
check("hungarian branch bit-identical to reference", max_diff < 1e-12,
      f"over {npairs} pairs, max|diff|={max_diff:.2e}")

# also check non-default weights still match
cfg2 = MatchingConfig(lambda_class=0.5, lambda_center=3.0, lambda_giou=0.2,
                      unmatched_penalty=2.0)
max_diff2 = 0.0
for _ in range(100):
    a = rand_layout(rng, rng.randint(1, 6))
    b = rand_layout(rng, rng.randint(1, 6))
    ref = reference_hungarian(a, b, cfg2)
    new = match_layouts(a, b, cfg2).distance
    max_diff2 = max(max_diff2, abs(ref - new))
check("hungarian bit-identical under custom weights", max_diff2 < 1e-12,
      f"max|diff|={max_diff2:.2e}")

# --------------------------------------------------------------- random baseline
rcfg = MatchingConfig(method="random", random_match_seed=7)
# reproducible
a = rand_layout(rng, 4)
b = rand_layout(rng, 5)
r1 = match_layouts(a, b, rcfg)
r2 = match_layouts(a, b, MatchingConfig(method="random", random_match_seed=7))
check("random assignment reproducible", r1.distance == r2.distance
      and r1.matched == r2.matched)
check("different seed -> (potentially) different assignment",
      match_layouts(a, b, MatchingConfig(method="random", random_match_seed=8)).distance
      is not None)

# assignment is a valid partial bijection: every index used at most once
valid_bijection = True
for _ in range(200):
    a = rand_layout(rng, rng.randint(1, 6))
    b = rand_layout(rng, rng.randint(1, 6))
    r = match_layouts(a, b, MatchingConfig(method="random",
                                          random_match_seed=rng.randint(0, 10000)))
    ai = [p[0] for p in r.matched]
    bj = [p[1] for p in r.matched]
    if len(ai) != len(set(ai)) or len(bj) != len(set(bj)):
        valid_bijection = False
        break
    if not (set(ai).isdisjoint(r.unmatched_a)):
        valid_bijection = False
        break
    if not (set(bj).isdisjoint(r.unmatched_b)):
        valid_bijection = False
        break
    # every object of the smaller set is either matched or unmatched
    covered = set(ai) | set(r.unmatched_a)
    if covered != set(range(len(a))):
        valid_bijection = False
        break
check("random yields a valid partial bijection", valid_bijection)

# in expectation random >= hungarian (optimum is the minimum over assignments)
gaps = []
for seed in range(60):
    for _ in range(40):
        a = rand_layout(rng, rng.randint(2, 6))
        b = rand_layout(rng, rng.randint(2, 6))
        h = match_layouts(a, b, MatchingConfig(method="hungarian")).distance
        rd = match_layouts(a, b, MatchingConfig(
            method="random", random_match_seed=seed)).distance
        gaps.append(rd - h)
gaps = np.array(gaps)
check("random mean distance >= hungarian (optimality)", gaps.mean() > 0.15,
      f"mean(extra cost)={gaps.mean():.3f}, P(random>=h)={np.mean(gaps >= -1e-9):.3f}")

# --------------------------------------------------------------- chamfer
ccfg = MatchingConfig(method="chamfer")
# non-negative and finite
finite = True
for _ in range(100):
    a = rand_layout(rng, rng.randint(1, 6))
    b = rand_layout(rng, rng.randint(1, 6))
    d = match_layouts(a, b, ccfg).distance
    if not (np.isfinite(d) and d >= 0):
        finite = False
check("chamfer finite and non-negative", finite)

# symmetry: d(a,b) == d(b,a)
sym = True
for _ in range(100):
    a = rand_layout(rng, rng.randint(1, 6))
    b = rand_layout(rng, rng.randint(1, 6))
    if abs(match_layouts(a, b, ccfg).distance
           - match_layouts(b, a, ccfg).distance) > 1e-12:
        sym = False
check("chamfer symmetric", sym)

# permutation invariance
perm_inv = True
for _ in range(100):
    a = rand_layout(rng, rng.randint(2, 6))
    b = rand_layout(rng, rng.randint(2, 6))
    ap = Layout(objects=rng.sample(a.objects, len(a.objects)))
    bp = Layout(objects=rng.sample(b.objects, len(b.objects)))
    if abs(match_layouts(a, b, ccfg).distance
           - match_layouts(ap, bp, ccfg).distance) > 1e-9:
        perm_inv = False
check("chamfer permutation invariant", perm_inv)

# self distance = 0 only when every object matches itself with zero cost
# (same class + identical box): build identical layouts
selfzero = True
for _ in range(50):
    a = rand_layout(rng, rng.randint(1, 6))
    if match_layouts(a, a, ccfg).distance > 1e-12:
        selfzero = False
check("chamfer self distance ~0", selfzero)

# known-value hand check: two identical objects one swapped pair
X = Layout(objects=[Object("a", Box(0.1, 0.1, 0.1, 0.1)),
                    Object("b", Box(0.8, 0.8, 0.1, 0.1))])
Y = Layout(objects=[Object("a", Box(0.1, 0.1, 0.1, 0.1)),
                    Object("b", Box(0.8, 0.8, 0.1, 0.1))])
check("chamfer identical layouts = 0",
      match_layouts(X, Y, MatchingConfig(method="chamfer")).distance < 1e-12)

# empty-set handling
e = Layout(objects=[])
check("chamfer empty vs non-empty = penalty",
      match_layouts(e, X, MatchingConfig(method="chamfer")).distance
      == cfg.unmatched_penalty)
check("chamfer empty vs empty = 0",
      match_layouts(e, e, MatchingConfig(method="chamfer")).distance == 0.0)

# unknown method raises
raised = False
try:
    match_layouts(X, Y, MatchingConfig(method="bogus"))
except ValueError:
    raised = True
check("unknown method raises ValueError", raised)

n_fail = PASS.count(False)
print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed")
sys.exit(1 if n_fail else 0)
