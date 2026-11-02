"""Composite layout distance: matching + count + relation terms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .matching import MatchingConfig, match_layouts
from .relations import relation_histogram
from .representation import Layout


@dataclass
class DistanceConfig:
    matching: MatchingConfig = None  # type: ignore[assignment]
    alpha: float = 1.0   # matched object distance
    gamma: float = 0.5   # count difference
    eta: float = 1.0     # relation histogram distance
    zeta: float = 0.0    # pairwise displacement-field term over matched pairs
    classes: Optional[List[str]] = None  # vocabulary for the relation histogram


def __post_init__(self):
    if self.matching is None:
        self.matching = MatchingConfig()


DistanceConfig.__post_init__ = __post_init__  # dataclass default factory workaround


def _displacement_field(a: Layout, b: Layout, res) -> float:
    """Mean pairwise relative-displacement disagreement over matched objects.

    For two matched objects (u->u', v->v'), compares the query-side relative
    position (p_u - p_v) with the candidate-side (p_u' - p_v'); the mean
    L2 disagreement over all unordered pairs, normalized by 2*sqrt(2) (the
    diameter of a unit box's displacement difference), keeping the term
    in [0, 1]. Returns 0 for fewer than two matched pairs.
    """
    pairs = [(i, j) for i, j in res.matched]
    if len(pairs) < 2:
        return 0.0
    ba, bb = a.boxes(), b.boxes()
    disp = []
    for i, j in pairs:
        ca = (ba[i][0] + ba[i][2] / 2.0, ba[i][1] + ba[i][3] / 2.0)
        cb = (bb[j][0] + bb[j][2] / 2.0, bb[j][1] + bb[j][3] / 2.0)
        disp.append((ca[0] - cb[0], ca[1] - cb[1]))
    n = len(disp)
    s = 0.0
    for x in range(n):
        for y in range(x + 1, n):
            dx = disp[x][0] - disp[y][0]
            dy = disp[x][1] - disp[y][1]
            s += float(np.hypot(dx, dy)) / (2.0 * np.sqrt(2.0))
    return s / (n * (n - 1) / 2.0)


def layout_distance(a: Layout, b: Layout, cfg: Optional[DistanceConfig] = None) -> Dict[str, float]:
    """Decomposed distance between two layouts (all terms in [0, ~3])."""
    cfg = cfg or DistanceConfig()
    res = match_layouts(a, b, cfg.matching)

    d_count = abs(len(a) - len(b)) / max(len(a), len(b), 1)

    d_rel = 0.0
    if cfg.eta > 0 and cfg.classes is not None and (len(a) > 1 or len(b) > 1):
        ha = relation_histogram(a, cfg.classes)
        hb = relation_histogram(b, cfg.classes)
        d_rel = float(np.abs(ha - hb).sum() / 2.0)  # total variation

    d_disp = _displacement_field(a, b, res) if cfg.zeta > 0 else 0.0

    total = (cfg.alpha * res.distance + cfg.gamma * d_count + cfg.eta * d_rel
             + cfg.zeta * d_disp)
    return {
        "total": total,
        "match": res.distance,
        "count": d_count,
        "relation": d_rel,
        "displacement": d_disp,
        "n_unmatched": float(len(res.unmatched_a) + len(res.unmatched_b)),
    }
