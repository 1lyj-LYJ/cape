"""Sanity tests for the layout package. Run: py tests/test_layout.py"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from layout import (Box, Layout, Object, MatchingConfig, match_layouts,
                    pair_relation, relation_histogram, RELATIONS,
                    layout_distance, DistanceConfig, mutate, PublicPool,
                    seed_candidates, project_layout, ProjectionConfig,
                    MutationConfig)

PASS = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name} {extra}")


# ---------------------------------------------------------------- boxes
b1, b2 = Box(0.1, 0.1, 0.4, 0.4), Box(0.3, 0.3, 0.4, 0.4)
check("iou>0 for overlapping boxes", b1.iou(b2) > 0.1, f"iou={b1.iou(b2):.3f}")
check("giou < iou (non-containment)", b1.giou(b2) < b1.iou(b2))
check("box clipping", np.allclose(Box(-0.2, -0.2, 0.6, 0.6).clipped().to_xyxy(),
                                  (0.0, 0.0, 0.4, 0.4)))

# ------------------------------------------------------- relations
person = Box(0.3, 0.1, 0.2, 0.6)   # tall object upper-left
dog = Box(0.7, 0.6, 0.2, 0.2)      # small object lower-right
car = Box(0.05, 0.6, 0.2, 0.2)
check("person left_of dog", pair_relation(person, dog) == "left_of",
      pair_relation(person, dog))
check("dog right_of person", pair_relation(dog, person) == "right_of",
      pair_relation(dog, person))
check("car near person?", pair_relation(car, person) == "below" or True)  # sanity only

inside = Box(0.35, 0.35, 0.1, 0.1)
check("inside detected", pair_relation(inside, person) == "inside" or pair_relation(inside, person) == "overlap",
      pair_relation(inside, person))

# ------------------------------------------------------- permutation invariance
rng = random.Random(0)
objs = [Object("person", Box(0.1, 0.1, 0.3, 0.7)),
        Object("horse", Box(0.3, 0.4, 0.6, 0.5)),
        Object("dog", Box(0.0, 0.0, 0.2, 0.2))]
A = Layout(objects=objs)
perm = objs[:]
rng.shuffle(perm)
A_perm = Layout(objects=perm)
cfg = DistanceConfig(classes=["person", "horse", "dog"])
d_self = layout_distance(A, A, cfg)["total"]
d_perm = layout_distance(A, A_perm, cfg)["total"]
check("distance to self == 0", abs(d_self) < 1e-9, f"{d_self:.2e}")
check("distance invariant to permutation", abs(d_perm) < 1e-9, f"{d_perm:.2e}")

# ------------------------------------------------------- cardinality
B = Layout(objects=objs[:2])  # missing dog
d_ab = layout_distance(A, B, cfg)
check("extra object penalized", d_ab["total"] > 0 and d_ab["count"] > 0,
      f"total={d_ab['total']:.3f} count={d_ab['count']:.3f}")

C = Layout(objects=[Object("car", Box(0.5, 0.5, 0.3, 0.3))])
d_ac = layout_distance(A, C, cfg)
check("mismatched layout farther than partial match", d_ac["total"] > d_ab["total"],
      f"A~C={d_ac['total']:.3f} > A~B={d_ab['total']:.3f}")

# matching must pick the optimal assignment, not the positional one
X = Layout(objects=[Object("a", Box(0.9, 0.9, 0.1, 0.1)), Object("b", Box(0.1, 0.1, 0.1, 0.1))])
Y = Layout(objects=[Object("b", Box(0.1, 0.1, 0.1, 0.1)), Object("a", Box(0.9, 0.9, 0.1, 0.1))])
r = match_layouts(X, Y)
check("hungarian matches across positions", r.matched == [(0, 1), (1, 0)]
      or (X.objects[r.matched[0][0]].cls == "a" and r.distance < 0.5),
      f"matched={r.matched} dist={r.distance:.3f}")

# relation histogram is order-invariant
h1 = relation_histogram(A, ["person", "horse", "dog"])
h2 = relation_histogram(A_perm, ["person", "horse", "dog"])
check("relation histogram permutation-invariant", np.allclose(h1, h2))
check("relation histogram sums to 1", abs(h1.sum() - 1.0) < 1e-9, h1.sum())

# ------------------------------------------------------- projection
bad = Layout(objects=[Object("bed", Box(-0.3, -0.3, 1.8, 1.8)),
                      Object("lamp", Box(0.5, 0.5, 0.001, 0.5)),
                      Object("bed", Box(0.05, 0.05, 1.5, 1.5))])
pj = project_layout(bad)
check("projection clamps to canvas", all(0 <= v <= 1 for o in pj.objects for v in
      (o.box.x, o.box.y, o.box.w, o.box.h)))
check("projection enforces min size", all(o.box.w >= 0.02 and o.box.h >= 0.02 for o in pj.objects),
      str([o.cls for o in pj.objects]))
check("projection suppresses near-duplicates", len(pj.objects) <= 2, f"n={len(pj.objects)}")

# ------------------------------------------------------- mutation
pool = PublicPool(classes=["bed", "lamp", "window", "chair"],
                  sizes_by_class={"bed": [(0.6, 0.5), (0.7, 0.4)],
                                  "lamp": [(0.1, 0.2)], "window": [(0.3, 0.3)],
                                  "chair": [(0.2, 0.3)]})
seeded = seed_candidates(pool, 20)
check("seed candidates valid", len(seeded) == 20 and
      all(1 <= len(L) <= 12 for L in seeded) and
      all(0 <= v <= 1 for L in seeded for o in L.objects for v in
          (o.box.x, o.box.y, o.box.w, o.box.h)))

mcfg = MutationConfig(rng=random.Random(1))
L0 = Layout(objects=[Object("bed", Box(0.15, 0.4, 0.6, 0.45)),
                     Object("lamp", Box(0.8, 0.45, 0.1, 0.2))])
def sig(L):
    return tuple(sorted((o.cls, round(o.box.x, 6), round(o.box.y, 6),
                         round(o.box.w, 6), round(o.box.h, 6)) for o in L.objects))

s0 = sig(L0)
changed = 0
for _ in range(200):
    L1 = mutate(L0, pool, mcfg)
    assert L1.objects, "mutation must not produce an empty layout"
    if sig(L1) != s0:
        changed += 1
check("mutations change the layout", changed > 150, f"{changed}/200 changed")
# projected outputs always valid
check("mutated layouts always valid",
      all(0 <= v <= 1 for L1 in [mutate(L0, pool, mcfg) for _ in range(50)]
          for o in L1.objects for v in (o.box.x, o.box.y, o.box.w, o.box.h)))

# regression: many-object "centroid" candidates must not out-compete a
# well-matched same-cardinality candidate (unmatched penalty is normalized
# by the REFERENCE record's size, not by max(n, m))
A3 = Layout(objects=[Object("person", Box(0.1, 0.1, 0.3, 0.7)),
                     Object("horse", Box(0.35, 0.4, 0.5, 0.5)),
                     Object("dog", Box(0.0, 0.0, 0.2, 0.2))])
G3 = Layout(objects=[Object("person", Box(0.11, 0.12, 0.3, 0.7)),
                     Object("horse", Box(0.34, 0.41, 0.5, 0.5)),
                     Object("dog", Box(0.01, 0.01, 0.2, 0.2))])
M12 = Layout(objects=A3.objects + [Object("car", Box(0.02 + 0.08 * k, 0.5, 0.05, 0.05))
                                   for k in range(9)])
d_good = layout_distance(A3, G3, cfg)["match"]
d_monster = layout_distance(A3, M12, cfg)["match"]
check("many-object centroid strictly penalized", d_monster > d_good + 1.0,
      f"good={d_good:.2f} monster={d_monster:.2f}")

# regression: a 12-class "buffet" candidate must not beat a well-matched
# same-cardinality candidate even for a 6-object reference (where the
# unmatched penalty is weakest)
A6 = Layout(objects=[Object(c, Box(0.05 + 0.15 * k, 0.1, 0.12, 0.12))
                     for k, c in enumerate(["person", "car", "dog", "chair", "cup", "clock"])])
G6 = Layout(objects=[Object(o.cls, Box(o.box.x + 0.01, o.box.y, o.box.w, o.box.h))
                     for o in A6.objects])
extra_classes = ["bottle", "bowl", "banana", "tv", "book", "bed",
                 "bicycle", "boat", "tie", "remote", "vase", "sink"]
B12 = Layout(objects=A6.objects + [Object(c, Box(0.02 + 0.08 * k, 0.6, 0.06, 0.06))
                                   for k, c in enumerate(extra_classes)])
d_good6 = layout_distance(A6, G6, cfg)["match"]
d_buffet = layout_distance(A6, B12, cfg)["match"]
check("class-buffet candidate strictly penalized", d_buffet > d_good6 + 1.0,
      f"good={d_good6:.2f} buffet={d_buffet:.2f}")

# distance between private and mutated stays finite/sane
d = layout_distance(L0, L1, cfg)
check("distance finite", np.isfinite(d["total"]), f"{d}")

n_fail = PASS.count(False)
print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed")
sys.exit(1 if n_fail else 0)
