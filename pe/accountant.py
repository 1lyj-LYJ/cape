"""Minimal RDP accountant for T rounds of Gaussian-mechanism voting.

Each round releases one histogram with L2 sensitivity 1 and Gaussian
noise sigma_t. The RDP of the Gaussian mechanism at order alpha is
    eps_rdp(alpha) = alpha / (2 * sigma_t^2)
Composition over rounds is additive in RDP (independent mechanisms).
Conversion to (eps, delta) is via the standard optimal-alpha bound:
    eps(delta) = min_alpha [ sum_t alpha/(2 sigma_t^2) + log(1/delta)/(alpha-1) ]

No subsampling amplification is assumed (each private record participates
in every round's vote — safe and simple). If later we sample records per
round, add Poisson subsampling RDP here.
"""
from __future__ import annotations

import math
from typing import List, Sequence


def rdp_gaussian(alpha: float, sigma: float) -> float:
    return alpha / (2.0 * sigma * sigma)


def compose_rdp(sigmas: Sequence[float], alpha: float) -> float:
    return sum(rdp_gaussian(alpha, s) for s in sigmas)


def epsilon(sigmas: Sequence[float], delta: float) -> float:
    """Total (eps, delta) after all rounds:
    eps = min_alpha [ sum_t alpha/(2 sigma_t^2) + log(1/delta)/(alpha-1) ]."""
    if not sigmas:
        return 0.0
    best = float("inf")
    alphas = [1.01 + 0.01 * i for i in range(2000)] + list(range(21, 257))
    for a in alphas:
        e = sum(a / (2.0 * s * s) for s in sigmas) + math.log(1.0 / delta) / (a - 1.0)
        best = min(best, e)
    return best


def sigma_for_epsilon(eps_total: float, delta: float, rounds: int) -> float:
    """Invert the accountant: given a TOTAL budget eps over T identical
    Gaussian releases, find the per-round sigma. (Never set eps per round!)
    Bisection on sigma — eps(sigma) is monotone decreasing."""
    lo, hi = 0.01, 10000.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if epsilon([mid] * rounds, delta) > eps_total:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    # sanity: T=10 rounds, sigma=100, delta=1e-5 -> eps should be ~0.15
    for T in (1, 10, 30):
        e = epsilon([100.0] * T, 1e-5)
        print(f"T={T:3d} sigma=100 delta=1e-5 -> eps={e:.3f}")
    print()
    for eps_t in (1.0, 2.0, 5.0, 10.0):
        s = sigma_for_epsilon(eps_t, 1e-5, 12)
        print(f"eps_total={eps_t:5.1f} T=12 delta=1e-5 -> per-round sigma={s:8.2f}")
