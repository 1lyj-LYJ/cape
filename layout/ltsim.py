"""LTSim layout distance (Otani et al., arXiv:2407.12356) as a PE support geometry.

The measure is an optimal-transport dissimilarity between two layouts:

    mu(e_i, e_j) = 1 - (delta_bbox + delta_label) / 2
    delta_bbox   = (1 + GIoU(b_i, b_j)) / 2
    delta_label  = 1 if c_i == c_j else 0

    EMD(L, L')   = min_{gamma >= 0} sum_ij gamma_ij mu(e_i, e_j)
                   s.t. row sums 1/m, column sums 1/n

LTSim itself maps EMD to a similarity via exp(-EMD / sigma); since exp is
monotone, nearest-candidate assignment under LTSim and under EMD is identical,
so PE uses the EMD value directly as the dissimilarity.
"""
from __future__ import annotations

import os

# POT imports PyTorch eagerly when it is installed; with 8 spawn workers that
# exhausts the Windows paging file. The transport here is tiny and purely
# numpy, so disable the optional backends before importing POT.
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")

import numpy as np

from .representation import Box, Layout


def ltsim_cost_matrix(a: Layout, b: Layout) -> np.ndarray:
    ca, cb = a.classes(), b.classes()
    ba, bb = a.boxes(), b.boxes()
    m, n = len(ca), len(cb)
    M = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        bi = Box(*ba[i])
        for j in range(n):
            giou = bi.giou(Box(*bb[j]))
            d_bbox = (1.0 + giou) / 2.0
            d_label = 1.0 if ca[i] == cb[j] else 0.0
            M[i, j] = 1.0 - (d_bbox + d_label) / 2.0
    return M


def ltsim_distance(a: Layout, b: Layout, reg: float = 1e-3) -> float:
    """Exact LTSim transport cost in [0, 1]."""
    m, n = len(a.objects), len(b.objects)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    import ot

    M = ltsim_cost_matrix(a, b)
    wa = np.full(m, 1.0 / m)
    wb = np.full(n, 1.0 / n)
    try:
        return float(ot.emd2(wa, wb, M))
    except Exception:                       # pragma: no cover
        return float(ot.sinkhorn2(wa, wb, M, reg))
