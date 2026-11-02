"""DP vote release: Gaussian noisy histogram over candidates.

Privacy units and adjacency (must be stated precisely in the paper):
  - One private IMAGE is one privacy record.
  - Adjacency = add/remove of one image (NOT replace-one). Under
    add/remove adjacency, a neighboring dataset changes the vote vector
    by ±e_j for a single candidate j, so the L2 sensitivity of the whole
    vote vector is Delta_2 = 1. (Under replace-one adjacency it would be
    sqrt(2) — we do not use that definition.)
  - The Gaussian mechanism with std sigma therefore applies to the
    vector as a whole — NOT per bbox, NOT per class. No bbox-level DP
    noise, ever.
"""
from __future__ import annotations

import numpy as np


def noisy_votes(true_votes: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to the vote histogram (add/remove adjacency, L2 sens = 1)."""
    return true_votes + rng.normal(0.0, sigma, size=true_votes.shape)


def sigma_from_epsilon(epsilon: float, delta: float) -> float:
    """Calibrate sigma for a single release of the vote histogram.

    Uses the classic Gaussian-machine conversion (Balle & Wang, 2018)
    for pure-DP -> (eps, delta):  sigma >= sqrt(2 log(1.25/delta)) / eps.
    Conservative and simple; the RDP accountant in accountant.py handles
    multi-round composition properly.
    """
    import math
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
