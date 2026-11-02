# -*- coding: utf-8 -*-
"""Layout-level cached relation feature (phi_rel), extracted from
layout.relations.relation_histogram's semantics so that the composite distance
is bit-identical, but computed ONCE per layout instead of once per pair.

Motivation (measured): relation_histogram allocates a dense (C, R, C) =
80*7*80 = 44,800-float array per call.  layout_distance calls it twice per pair,
and the vote query evaluates N*K = 2.86e6 pairs per round -> ~1 TB of allocation
traffic per round.  Hoisting it to a layout-level feature turns

    O(N*M * C_extract)        into        O((N+M) * C_extract) + O(N*M * d_r)

which is what makes the relation-enabled query tractable at all (and is the
implementation a reviewer would expect if the term is ever adopted).

phi_rel(L) is returned as a sparse dict {(ci, rel, cj): normalized_mass}, which
matches relation_histogram up to the representation: that function returns a
dense normalized (C,R,C) array, and the distance uses 0.5*|ha - hb|_1, i.e. the
total variation between the two normalized distributions.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from layout.relations import RELATIONS, pair_relation
from layout.representation import Layout

_RI = {r: k for k, r in enumerate(RELATIONS)}


def phi_rel(layout: Layout, classes: Sequence[str]) -> Dict[Tuple[int, int, int], float]:
    """Sparse normalized relation histogram of one layout (layout-level feature)."""
    if len(layout.objects) < 2:
        return {}
    ci = {c: k for k, c in enumerate(classes)}
    d: Dict[Tuple[int, int, int], float] = {}
    objs = layout.objects
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            rel = pair_relation(a.box, b.box)
            ia, ib = ci.get(a.cls), ci.get(b.cls)
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
            k = (ia, _RI[rel], ib)
            d[k] = d.get(k, 0.0) + 1.0
    s = sum(d.values())
    if s > 0:
        d = {k: v / s for k, v in d.items()}
    return d


def rel_tv(da: Dict, db: Dict) -> float:
    """Total variation between two sparse normalized relation histograms.

    Equals 0.5 * sum_k |ha_k - hb_k| over the union of keys, i.e. exactly what
    layout_distance computes as its ``relation`` term.
    """
    if not da and not db:
        return 0.0
    tot = 0.0
    for k, v in da.items():
        tot += abs(v - db.get(k, 0.0))
    for k, v in db.items():
        if k not in da:
            tot += abs(v)
    return 0.5 * tot
