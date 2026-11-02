"""Exact optimal-transport (earth mover's distance) layout distance.

Baseline for the correspondence-aware distance.  The ONLY difference from
``layout.matching.match_layouts`` is the assignment rule:

  * ``match_layouts``  : hard one-to-one assignment (Hungarian), optional
                         count-mismatch penalty;
  * ``emd_layout``     : soft transport between two normalized object
                         distributions (each layout summed to mass 1), so
                         object count is *normalized away*;
  * ``emd_layout_cnt`` : the same transport plus the identical count penalty
                         used by the paper, i.e. OT given its strongest fair
                         form.

Ground cost is the SAME per-object cost as the paper (class indicator +
GIoU/center/size), so the comparison isolates the assignment rule, not the
feature design.  Exact transport is solved with POT's network simplex.
"""
from __future__ import annotations

import numpy as np

from .matching import MatchingConfig, cost_matrix


def _cost(a, b, cfg: MatchingConfig) -> np.ndarray:
    return cost_matrix(a, b, cfg)


def _count_penalty(m: int, n: int) -> float:
    if max(m, n) == 0:
        return 0.0
    return abs(m - n) / float(max(m, n))


def emd_layout(a, b, cfg: MatchingConfig, lam_cnt: float = 0.0,
               reg: float = 1e-3) -> float:
    """Normalized EMD; ``lam_cnt > 0`` adds the paper's count term."""
    m, n = len(a.objects), len(b.objects)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        # mass cannot be transported at all: charge the maximum unit cost
        return 1.0 + lam_cnt
    import ot

    A = _cost(a, b, cfg)
    wa = np.full(m, 1.0 / m)
    wb = np.full(n, 1.0 / n)
    try:
        val = float(ot.emd2(wa, wb, A))
    except Exception:                      # pragma: no cover
        val = float(ot.sinkhorn2(wa, wb, A, reg))
    return val + lam_cnt * _count_penalty(m, n)
