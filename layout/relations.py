"""Geometric relations between object pairs, derived from boxes alone.

A relation signature of a layout is the multiset of directed relations
over all ordered object pairs. The signature is permutation invariant
because it indexes pairs by (class_i, class_j, relation) rather than
by object position.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .representation import Box, Layout

RELATIONS = ("left_of", "right_of", "above", "below", "overlap", "inside", "near")

# overlap ratio at which two boxes are considered overlapping
_OVERLAP_IOU = 0.15
# relative distance (w.r.t. the larger box diagonal) under which boxes are "near"
_NEAR_DIST = 0.15
# relative offset (w.r.t. the larger box size) needed for a directional relation
_DIR_MARGIN = 0.15


def pair_relation(a: Box, b: Box) -> str:
    """Relation of ``a`` with respect to ``b`` (directed)."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    # containment: how much of the smaller-relation candidate sits inside b
    contain_a = inter / max(a.area, 1e-9)
    contain_b = inter / max(b.area, 1e-9)
    if contain_a >= 0.8 and a.area <= b.area:
        return "inside"
    if a.iou(b) >= _OVERLAP_IOU or contain_b >= 0.8:
        return "overlap"

    dx = a.cx - b.cx
    dy = a.cy - b.cy
    scale = max(max(a.w, a.h), max(b.w, b.h))
    # near: centers close relative to object size, but boxes don't overlap
    if np.hypot(dx, dy) <= _NEAR_DIST * scale * 4.0:
        return "near"

    if abs(dx) >= abs(dy):
        return "left_of" if dx < 0 else "right_of"
    return "above" if dy < 0 else "below"


def relation_signature(layout: Layout) -> Counter:
    """Counter of (class_i, relation, class_j) triples over ordered pairs.

    Ordered pairs i < j in the current (arbitrary) object order, but the
    triple is keyed by class names so the signature is order-independent.
    """
    sig: Counter = Counter()
    objs = layout.objects
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            rel = pair_relation(a.box, b.box)
            sig[(a.cls, rel, b.cls)] += 1
    return sig


def relation_histogram(layout: Layout, classes: List[str]) -> np.ndarray:
    """Fixed-size relation histogram vector for distribution comparisons.

    Shape (C, R, C) with C = len(classes), R = len(RELATIONS); entry
    [ci, r, cj] counts objects of class ci standing in relation r to
    objects of class cj (direction resolved canonically: left_of/above
    are stored with the object that is left/above first).
    """
    cls_idx = {c: k for k, c in enumerate(classes)}
    hist = np.zeros((len(classes), len(RELATIONS), len(classes)), dtype=np.float64)
    rel_idx = {r: k for k, r in enumerate(RELATIONS)}
    for i in range(len(layout.objects)):
        for j in range(i + 1, len(layout.objects)):
            a, b = layout.objects[i], layout.objects[j]
            rel = pair_relation(a.box, b.box)
            ia, ib = cls_idx.get(a.cls), cls_idx.get(b.cls)
            if ia is None or ib is None:
                continue
            if rel in ("left_of", "right_of"):
                if rel == "right_of":
                    ia, ib = ib, ia
                rel = "left_of"
            elif rel in ("above", "below"):
                if rel == "below":
                    ia, ib = ib, ia
                rel = "above"
            hist[ia, rel_idx[rel], ib] += 1.0
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist
