"""Layout-distribution metrics: how well does a synthetic layout set
match the private layout set? All metrics compare *distributions*, never
individual layouts, and all are permutation invariant.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from layout.relations import relation_histogram
from layout.representation import Layout


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.abs(p - q).sum() / 2.0)


def _norm(hist: Counter, keys: Sequence) -> np.ndarray:
    v = np.array([hist.get(k, 0) for k in keys], dtype=np.float64)
    return v / v.sum() if v.sum() > 0 else v


def count_distribution(layouts: List[Layout], max_count: int = 8) -> np.ndarray:
    c = Counter(min(len(l), max_count) for l in layouts)
    return _norm(c, range(1, max_count + 1))


def class_distribution(layouts: List[Layout], classes: List[str]) -> np.ndarray:
    c = Counter(o.cls for l in layouts for o in l.objects)
    return _norm(c, classes)


def relation_distribution(layouts: List[Layout], classes: List[str]) -> np.ndarray:
    """Flattened (C, R, C) relation histograms, normalized jointly."""
    from layout.relations import RELATIONS as _REL
    if not layouts:
        return np.zeros(len(classes) * len(_REL) * len(classes))
    h = np.zeros((len(classes), len(_REL), len(classes)), dtype=np.float64)
    for l in layouts:
        h += relation_histogram(l, classes)
    return (h / h.sum()).ravel() if h.sum() > 0 else h.ravel()


def box_center_histogram(layouts: List[Layout], bins: int = 5) -> np.ndarray:
    """Joint (cx, cy) histogram over all objects."""
    h = np.zeros((bins, bins), dtype=np.float64)
    for l in layouts:
        for o in l.objects:
            i = min(int(o.box.cx * bins), bins - 1)
            j = min(int(o.box.cy * bins), bins - 1)
            h[i, j] += 1
    return (h / h.sum()).ravel() if h.sum() > 0 else h.ravel()


def box_size_histogram(layouts: List[Layout], bins: int = 5) -> np.ndarray:
    """Joint (w, h) histograms on a log2 grid (sizes span two decades)."""
    h = np.zeros((bins, bins), dtype=np.float64)
    for l in layouts:
        for o in l.objects:
            iw = min(int(np.log2(max(o.box.w, 1e-3) / 1e-3) / (np.log2(1000) / bins)), bins - 1)
            ih = min(int(np.log2(max(o.box.h, 1e-3) / 1e-3) / (np.log2(1000) / bins)), bins - 1)
            h[iw, ih] += 1
    return (h / h.sum()).ravel() if h.sum() > 0 else h.ravel()


def scale_distribution(layouts: List[Layout]) -> np.ndarray:
    return np.array([o.box.area for l in layouts for o in l.objects])


def w1_1d(a: Sequence[float], b: Sequence[float]) -> float:
    """1-Wasserstein between two empirical 1-D samples."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(_w1(np.sort(np.asarray(a)), np.sort(np.asarray(b))))


def _w1(sa: np.ndarray, sb: np.ndarray) -> float:
    n, m = len(sa), len(sb)
    allv = np.concatenate([sa, sb])
    cdfa = np.searchsorted(sa, allv, side="right") / n
    cdfb = np.searchsorted(sb, allv, side="right") / m
    return float(np.trapezoid(np.abs(cdfa - cdfb), allv))


def relation_marginal(layouts: List[Layout]) -> np.ndarray:
    """Distribution over the 7 relation types (class-marginalized)."""
    from layout.relations import RELATIONS, pair_relation
    c = Counter()
    for l in layouts:
        for i in range(len(l.objects)):
            for j in range(i + 1, len(l.objects)):
                c[pair_relation(l.objects[i].box, l.objects[j].box)] += 1
    return _norm(c, RELATIONS)


def layout_set_report(private: List[Layout], synthetic: List[Layout],
                      classes: List[str]) -> Dict[str, float]:
    """The full Gate-2 metric panel."""
    rep: Dict[str, float] = {}
    rep["count_TV"] = total_variation(count_distribution(private), count_distribution(synthetic))
    rep["class_TV"] = total_variation(class_distribution(private, classes),
                                      class_distribution(synthetic, classes))
    rep["relation_TV"] = total_variation(relation_distribution(private, classes),
                                         relation_distribution(synthetic, classes))
    rep["relation_marginal_TV"] = total_variation(relation_marginal(private),
                                                  relation_marginal(synthetic))
    rep["center_TV"] = total_variation(box_center_histogram(private), box_center_histogram(synthetic))
    rep["size_TV"] = total_variation(box_size_histogram(private), box_size_histogram(synthetic))
    rep["scale_W1"] = w1_1d(scale_distribution(private), scale_distribution(synthetic))
    rep["aspect_W1"] = w1_1d(
        [o.box.w / max(o.box.h, 1e-9) for l in private for o in l.objects],
        [o.box.w / max(o.box.h, 1e-9) for l in synthetic for o in l.objects])
    return rep
