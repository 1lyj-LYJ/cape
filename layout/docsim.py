# -*- coding: utf-8 -*-
"""DocSim layout similarity from READ (CVPR Workshop 2020).

DocSim assigns a weight to every same-type object pair, solves a maximum
weight bipartite matching, and averages the matched weights.  This module
adapts the published formula to the project's normalized xywh ``Layout``
representation.

Reference:
    Patil et al., READ: Recursive Autoencoders for Document Layout
    Generation, CVPRW 2020, Section 4.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .representation import Layout


def docsim_similarity(a: Layout, b: Layout, shape_constant: float = 2.0,
                      area_power: float = 0.5) -> float:
    """Return the DocSim similarity in [0, 1] (higher is more similar)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0

    weights = np.zeros((n, m), dtype=np.float64)
    for i, oa in enumerate(a.objects):
        for j, ob in enumerate(b.objects):
            if oa.cls != ob.cls:
                continue
            dc = math.hypot(oa.box.cx - ob.box.cx, oa.box.cy - ob.box.cy)
            ds = abs(oa.box.w - ob.box.w) + abs(oa.box.h - ob.box.h)
            area = min(oa.box.area, ob.box.area) ** area_power
            weights[i, j] = area * (2.0 ** (-dc - shape_constant * ds))

    # READ states that different-type boxes have zero weight and are
    # effectively unmatchable.  Pad the assignment with dummy zero-cost rows
    # and columns so the optimizer may leave objects unmatched instead of
    # forcing zero-weight cross-type edges into the denominator.
    size = n + m
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:n, :m] = -weights
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [weights[i, j] for i, j in zip(row_ind, col_ind)
               if i < n and j < m and weights[i, j] > 0.0]
    if not matched:
        return 0.0

    # The published metric averages the matched edge weights.
    return float(np.sum(matched) / len(matched))


def docsim_distance(a: Layout, b: Layout) -> float:
    """Distance form used by nearest-candidate private voting."""
    return 1.0 - docsim_similarity(a, b)
