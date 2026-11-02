"""Joint structural metrics for layout sets (beyond marginal histograms).

The six headline metrics are marginals: count of objects, class frequency,
marginal spatial-relation histogram, center/size/scale.  A method could match
all of them while getting *which classes co-occur* and *how class pairs are
positioned relative to each other* wrong.  These two joint metrics close that
gap:

  cooccurrence_TV  : total variation between the class-pair co-occurrence
                     matrices  P(c1, c2 both present in one layout);
  orientation_TV   : total variation between the class-pair relative-position
                     distributions  P(rel | c1, c2), rel in
                     {left, right, above, below, overlap}.

Both are computed directly on layout sets (no optimal matching, no reference to
the paper's distance), so they are independent of the training objective.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

RELS = ("left", "right", "above", "below", "overlap")


def _rel(a, b) -> str:
    """Relative position of object a w.r.t. object b (boxes are xywh)."""
    ax, ay = a.box.cx, a.box.cy
    bx, by = b.box.cx, b.box.cy
    dx, dy = ax - bx, ay - by
    dxr = dx / max(1e-9, (a.box.w + b.box.w) / 2.0)
    dyr = dy / max(1e-9, (a.box.h + b.box.h) / 2.0)
    if abs(dxr) < 0.5 and abs(dyr) < 0.5:
        return "overlap"
    if abs(dxr) >= abs(dyr):
        return "right" if dxr > 0 else "left"
    return "below" if dyr > 0 else "above"


def cooccurrence(layouts: Sequence, classes: Iterable[str]) -> np.ndarray:
    cls = list(classes)
    idx = {c: i for i, c in enumerate(cls)}
    M = np.zeros((len(cls), len(cls)), dtype=np.float64)
    for lay in layouts:
        present = sorted({o.cls for o in lay.objects if o.cls in idx})
        for a in present:
            for b in present:
                if a != b:
                    M[idx[a], idx[b]] += 1.0
    tot = float(len(layouts)) or 1.0
    return M / tot


def orientation(layouts: Sequence, classes: Iterable[str]) -> Dict[Tuple[str, str],
                                                                   np.ndarray]:
    cls = set(classes)
    acc: Dict[Tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(len(RELS)))
    for lay in layouts:
        objs = [o for o in lay.objects if o.cls in cls]
        for a in objs:
            for b in objs:
                if a is b or a.cls == b.cls:
                    continue
                i = RELS.index(_rel(a, b))
                acc[(a.cls, b.cls)][i] += 1.0
    out = {}
    for k, v in acc.items():
        s = v.sum()
        if s > 0:
            out[k] = v / s
    return out


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def joint_report(priv: Sequence, synth: Sequence,
                 classes: Iterable[str]) -> Dict[str, float]:
    cls = list(classes)
    cp = cooccurrence(priv, cls)
    cs = cooccurrence(synth, cls)
    # normalise the ordered-pair mass so TV is bounded in [0, 1]
    sp, ss = float(cp.sum()), float(cs.sum())
    cp_n = cp / sp if sp > 0 else cp
    cs_n = cs / ss if ss > 0 else cs
    op = orientation(priv, cls)
    os_ = orientation(synth, cls)
    keys = [k for k in op if k in os_ and op[k].sum() > 0 and os_[k].sum() > 0]
    w = np.array([cp_n[cls.index(k[0]), cls.index(k[1])] for k in keys]) \
        if keys else np.array([])
    if keys and w.sum() > 0:
        orient_tv = float(np.average([_tv(op[k], os_[k]) for k in keys],
                                     weights=w))
    else:
        orient_tv = float("nan")
    return {
        "cooccurrence_TV": _tv(cp_n, cs_n),
        "orientation_TV": orient_tv,
        "n_class_pairs": float(len(keys)),
    }
