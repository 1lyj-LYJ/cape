"""Permutation-invariant layout matching via the Hungarian algorithm.

The cost of matching two objects combines class disagreement, center
distance, size distance and (1 - GIoU). Unmatched objects pay a fixed
penalty. The resulting distance is invariant to object ordering and
defined for layouts of different cardinality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .representation import Layout


@dataclass
class MatchingConfig:
    lambda_class: float = 1.0      # penalty for class mismatch
    lambda_center: float = 2.0     # L2 distance between centers
    lambda_size: float = 1.0       # L1 distance between (w, h)
    lambda_giou: float = 1.0       # (1 - GIoU), in [0, 2]
    unmatched_penalty: float = 2.5  # penalty per unmatched object (normalized
    # by the reference record's size). Must exceed the typical per-pair
    # matched cost (~1-2) so that "class buffet" candidates — many diverse
    # objects that give every reference object a zero-class-mismatch
    # partner — never pay off despite their unmatched tail.

    # --- matching-method selection (for the correspondence baselines) ---
    # "hungarian" : optimal one-to-one assignment (the paper's method;
    #               the default keeps every legacy run bit-identical).
    # "random"    : a random one-to-one assignment on the SAME padded cost
    #               matrix (only the optimization is removed). This isolates
    #               whether the OPTIMALITY of the matching matters.
    # "chamfer"   : symmetric bidirectional nearest-neighbour distance
    #               (no one-to-one constraint); the standard Chamfer set
    #               distance on the same pairwise object cost.
    method: str = "hungarian"
    # Seed for the random-assignment baseline. The per-pair RNG is derived
    # deterministically from this seed and the two cardinalities so that the
    # comparison is reproducible and does not depend on object order.
    random_match_seed: int = 0


def object_cost(o1_cls: str, b1: np.ndarray, o2_cls: str, b2: np.ndarray,
                cfg: MatchingConfig) -> float:
    """b1, b2: xywh arrays of shape (4,)."""
    cls_cost = cfg.lambda_class * (0.0 if o1_cls == o2_cls else 1.0)
    c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
    c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
    center = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
    size = abs(b1[2] - b2[2]) + abs(b1[3] - b2[3])
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    from .representation import Box
    giou = Box(x1, y1, w1, h1).giou(Box(x2, y2, w2, h2))
    return cls_cost + cfg.lambda_center * center + cfg.lambda_size * size + cfg.lambda_giou * (1.0 - giou)


def cost_matrix(a: Layout, b: Layout, cfg: MatchingConfig) -> np.ndarray:
    ca, cb = a.classes(), b.classes()
    ba, bb = a.boxes(), b.boxes()
    n, m = len(ca), len(cb)
    C = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            C[i, j] = object_cost(ca[i], ba[i], cb[j], bb[j], cfg)
    return C


@dataclass
class MatchResult:
    distance: float          # total matched distance (mean over matched pairs)
    matched: List[Tuple[int, int]]  # index pairs (i in a, j in b)
    unmatched_a: List[int]
    unmatched_b: List[int]
    cost_matrix: np.ndarray


def _empty_result(cfg: MatchingConfig, n: int, m: int) -> MatchResult:
    """Distance when one (or both) layouts are empty."""
    if n == 0 and m == 0:
        return MatchResult(0.0, [], [], [], np.zeros((0, 0)))
    return MatchResult(
        cfg.unmatched_penalty,
        [],
        list(range(n)) if n else [],
        list(range(m)) if m else [],
        np.zeros((n, m)),
    )


def _finalize_padded_assignment(C, n, m, row_ind, col_ind, cfg):
    """Turn a K x K padded assignment (row_ind -> col_ind) into a MatchResult.

    Shared by the Hungarian and random-assignment baselines: indices inside
    the real n x m block are matched pairs; assignments touching padded
    dummy rows/columns are unmatched objects. The distance normalization is
    identical for both, so the ONLY thing that differs is the assignment.
    """
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
    dist = mean_cost + cfg.unmatched_penalty * n_unmatched / max(n, 1)
    return MatchResult(dist, matched, ua, ub, C)


def _match_hungarian(a: Layout, b: Layout, C: np.ndarray,
                     cfg: MatchingConfig) -> MatchResult:
    """Optimal one-to-one assignment on the padded cost matrix."""
    n, m = len(a), len(b)
    K = max(n, m)
    Cp = np.full((K, K), cfg.unmatched_penalty, dtype=np.float64)
    Cp[:n, :m] = C
    row_ind, col_ind = linear_sum_assignment(Cp)
    return _finalize_padded_assignment(C, n, m, row_ind, col_ind, cfg)


def _match_random(a: Layout, b: Layout, C: np.ndarray,
                  cfg: MatchingConfig) -> MatchResult:
    """Random one-to-one assignment on the SAME padded cost matrix.

    The cost matrix is identical to the Hungarian baseline; only the
    optimization is removed. A uniform random permutation of the K padded
    columns pairs each row with one column (real or dummy). The per-pair
    permutation RNG is keyed by the cardinalities and a fixed seed, so the
    result is reproducible and does not depend on object content or order.
    """
    n, m = len(a), len(b)
    K = max(n, m)
    # Distinct (but deterministic) permutation for each (n, m, seed) cell.
    rng = np.random.default_rng(
        cfg.random_match_seed * 1_000_003 + 7919 * (n + 1) + 31 * (m + 1))
    row_ind = np.arange(K)
    col_ind = rng.permutation(K)
    return _finalize_padded_assignment(C, n, m, row_ind, col_ind, cfg)


def _match_chamfer(a: Layout, b: Layout, C: np.ndarray,
                   cfg: MatchingConfig) -> MatchResult:
    """Symmetric Chamfer distance on the pairwise object cost.

    d = 0.5 * [ mean_i min_j C[i,j] + mean_j min_i C[i,j] ].
    Every object is matched to its nearest neighbour in the other set, so
    there is no one-to-one constraint and no explicit unmatched penalty;
    cardinality differences remain penalized by the count term in
    layout_distance. The returned MatchResult records both directed
    nearest-neighbour sets (a pair may appear twice) for diagnostics.
    """
    n, m = len(a), len(b)
    a_to_b = C.min(axis=1)          # nearest candidate object for each of a
    b_to_a = C.min(axis=0)          # nearest reference object for each of b
    dist = 0.5 * (float(a_to_b.mean()) + float(b_to_a.mean()))
    nn_b = C.argmin(axis=1)
    nn_a = C.argmin(axis=0)
    matched = [(int(i), int(nn_b[i])) for i in range(n)]
    matched += [(int(nn_a[j]), int(j)) for j in range(m)]
    return MatchResult(dist, matched, [], [], C)


def match_layouts(a: Layout, b: Layout, cfg: Optional[MatchingConfig] = None) -> MatchResult:
    """Layout distance under the configured object-correspondence method.

    All methods share the same pairwise object cost (class + center + size +
    GIoU). The returned distance for Hungarian/random is
        mean(matched costs) + unmatched_penalty * n_unmatched / max(n_a, 1)
    so layouts with different object counts are penalized while the value
    stays comparable. Chamfer uses the symmetric bidirectional nearest
    neighbour mean with no explicit unmatched penalty.
    """
    cfg = cfg or MatchingConfig()
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return _empty_result(cfg, n, m)

    C = cost_matrix(a, b, cfg)
    if cfg.method == "hungarian":
        return _match_hungarian(a, b, C, cfg)
    if cfg.method == "random":
        return _match_random(a, b, C, cfg)
    if cfg.method == "chamfer":
        return _match_chamfer(a, b, C, cfg)
    raise ValueError(f"unknown matching method: {cfg.method!r} "
                     "(expected 'hungarian', 'random' or 'chamfer')")
