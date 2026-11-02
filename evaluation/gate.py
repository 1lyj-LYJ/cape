"""Gate 2 pass criteria, fixed in advance (no tuning-until-it-looks-nice).

PASS requires, relative to the raw-public-seed baseline:
  - at least 5 of 6 distribution metrics improved (lower);
  - no metric catastrophically worse (ratio > 1.2 fails the gate).

The gated metrics: count_TV, class_TV, relation_marginal_TV, center_TV,
size_TV, scale_W1. (class-conditioned relation_TV is too sparse at smoke
scale to gate on; it is reported but not gated.)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

GATED = ["count_TV", "class_TV", "relation_marginal_TV", "center_TV", "size_TV", "scale_W1"]
CATASTROPHIC_RATIO = 1.2
MIN_IMPROVED = 5


def gate_verdict(ours: Dict[str, float], baseline: Dict[str, float]) -> Tuple[bool, List[dict]]:
    rows = []
    for m in GATED:
        base, cur = baseline[m], ours[m]
        rows.append({
            "metric": m,
            "baseline": round(base, 4),
            "ours": round(cur, 4),
            "ratio": round(cur / base, 3) if base > 0 else float("inf"),
            "improved": bool(cur < base),
            "catastrophic": bool(base > 0 and cur > CATASTROPHIC_RATIO * base),
        })
    n_improved = sum(r["improved"] for r in rows)
    n_catastrophic = sum(r["catastrophic"] for r in rows)
    passed = n_improved >= MIN_IMPROVED and n_catastrophic == 0
    return passed, rows
