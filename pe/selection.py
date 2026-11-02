"""Diversity-Preserving Private Evolution Selection.

Plain top-k selection by (noisy) votes collapses onto a few generic
vote-winning templates ("two persons"), which destroys the count / class
/ aspect coverage of the candidate population. The selection step is
therefore:

  1. **count-bucket stratification** — candidates are bucketed by their
     OWN object count (a public property of the candidate, never of the
     private data), and every bucket is guaranteed a minimum share of
     survival slots, so rare complex layouts cannot be out-competed;
  2. **greedy redundancy penalty** — after the guaranteed slots, the
     remaining slots are filled greedily by
         S_j = v_j - lambda * max_{k in S} sim(L_j, L_k)
     where v_j is the (noisy) vote score and the redundancy is the
     similarity to the ALREADY-SELECTED set. A high-vote template can
     still be selected, but not dozens of copies of it.

Both steps consume only (a) the DP-released noisy votes and (b) public
candidate layouts — they are deterministic post-processing of the DP
histogram release and cost zero additional privacy budget.

The similarity here is deliberately a CHEAP public proxy (class-multiset
Jaccard x center proximity), not the full Hungarian layout distance:
selection runs every round over the whole candidate pool.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from layout.representation import Layout

# buckets over the candidate's own object count
COUNT_BUCKETS: Tuple[Tuple[int, int], ...] = ((1, 2), (3, 4), (5, 6), (7, 10**9))


def count_bucket(n: int) -> int:
    for b, (lo, hi) in enumerate(COUNT_BUCKETS):
        if lo <= n <= hi:
            return b
    return len(COUNT_BUCKETS) - 1


@dataclass
class SelectionConfig:
    n_select: int = 240            # total parents returned
    n_elite: int = 60              # slots filled by pure votes (best templates)
    lambda_redundancy: float = 0.6  # weight of the redundancy penalty
    min_bucket_frac: float = 0.12   # flat guaranteed share per count bucket
    count_quotas: Optional[List[float]] = None
    """Optional per-bucket fractions of n_select allocated PROPORTIONALLY to
    the public prior's object-count distribution (public data). Makes the
    parent pool's count distribution track the prior instead of collapsing
    onto vote-winning cardinalities. Fractions should sum to <= ~0.6 so the
    remaining slots stay contestable."""
    rare_frac: float = 0.15         # guaranteed share for rare-class-dominant candidates
    strata_quotas: Optional[Dict[Tuple[int, int], float]] = None
    """Joint (count-bucket, class-decile) quotas as fractions of n_free,
    proportional to the PUBLIC prior's joint distribution (obtained by
    simulating the public generator). Makes the parent pool mirror the
    prior on BOTH cardinality and class-mix while votes decide within
    each stratum."""
    freq: Optional[dict] = None     # public class frequencies (counts)
    rare_threshold: float = 0.005   # public freq below this = rare class
    center_tau: float = 0.25        # center-distance scale in the cheap similarity


def _class_multiset(layout: Layout) -> Counter:
    return Counter(o.cls for o in layout.objects)


def _centroid(layout: Layout) -> np.ndarray:
    if not layout.objects:
        return np.zeros(2)
    return np.mean([[o.box.cx, o.box.cy] for o in layout.objects], axis=0)


def _dominant_aspect(layout: Layout) -> float:
    """Aspect (w/h) of the largest-area object — the layout's visual anchor."""
    if not layout.objects:
        return 1.0
    o = max(layout.objects, key=lambda o: o.box.area)
    return o.box.w / max(o.box.h, 1e-9)


class ClassTiers:
    """Public-frequency class tiers (never uses private statistics)."""

    def __init__(self, freq: Optional[dict], threshold: float):
        self.rare: set = set()
        self.decile: dict = {}
        if freq:
            total = float(sum(freq.values())) or 1.0
            self.rare = {c for c, n in freq.items() if n / total < threshold}
            ranked = sorted(freq, key=lambda c: -freq[c])
            for rank, c in enumerate(ranked):
                self.decile[c] = min(9, int(rank / max(1, len(ranked)) * 10))

    def tier(self, cls: str) -> int:
        return self.decile.get(cls, 9)

    def is_rare_dominant(self, layout: Layout) -> bool:
        if not layout.objects or not self.rare:
            return False
        dominant = Counter(o.cls for o in layout.objects).most_common(1)[0][0]
        return dominant in self.rare


def dominant_class(layout: Layout) -> Optional[str]:
    if not layout.objects:
        return None
    return Counter(o.cls for o in layout.objects).most_common(1)[0][0]


def stratum_key(layout: Layout, tiers: "ClassTiers") -> Tuple[int, int]:
    """Joint stratum: (object-count bucket, dominant-class frequency decile)."""
    return (count_bucket(len(layout)), tiers.tier(dominant_class(layout) or ""))


def prior_strata_quotas(pool, freq: Optional[dict], rare_threshold: float = 0.005,
                        n_sim: int = 10000, scale: float = 0.70,
                        seed: int = 777) -> Dict[Tuple[int, int], float]:
    """Simulate the PUBLIC generator (candidate seeding) and return joint
    strata quotas proportional to its (count x class-decile) profile.
    Purely public data; deterministic given the seed."""
    from layout.mutation import MutationConfig, seed_candidates
    tiers = ClassTiers(freq, rare_threshold)
    sims = seed_candidates(pool, n_sim, MutationConfig(rng=random.Random(seed)))
    tally: Counter = Counter(stratum_key(L, tiers) for L in sims)
    total = float(sum(tally.values())) or 1.0
    return {k: scale * v / total for k, v in tally.items()}


def cheap_similarity(a: Layout, b: Layout, fa: Tuple, fb: Tuple,
                     tau: float) -> float:
    """Cheap public proxy: class-multiset Jaccard x exp(-center distance / tau)."""
    ca, cb = fa[0], fb[0]
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    jac = inter / union if union > 0 else 0.0
    dist = float(np.linalg.norm(fa[1] - fb[1]))
    return jac * float(np.exp(-dist / tau))


class DiversitySelector:
    def __init__(self, cfg: Optional[SelectionConfig] = None):
        self.cfg = cfg or SelectionConfig()

    def __call__(self, votes: np.ndarray, candidates: List[Layout],
                 rng: random.Random) -> List[int]:
        cfg = self.cfg
        n = len(candidates)
        v = np.clip(np.asarray(votes, dtype=np.float64), 0.0, None)
        v_hat = v / v.max() if v.max() > 0 else np.zeros(n)

        feats: List[Tuple] = [(_class_multiset(c), _centroid(c)) for c in candidates]
        buckets: Dict[int, List[int]] = {}
        for i, c in enumerate(candidates):
            buckets.setdefault(count_bucket(len(c)), []).append(i)
        for b in buckets:
            rng.shuffle(buckets[b])

        selected: List[int] = []
        # incremental cache: max similarity of each candidate to the
        # already-selected set — redundancy(i) is then O(1) to read
        max_sim = np.zeros(n, dtype=np.float64)

        def greedy(pool: List[int], k: int) -> None:
            """Pick k indices from pool by S = v_hat - lambda * redundancy."""
            remaining = set(pool)
            for _ in range(min(k, len(pool))):
                if not remaining:
                    break
                best_i = max(remaining, key=lambda i: v_hat[i] - cfg.lambda_redundancy * max_sim[i])
                selected.append(best_i)
                remaining.discard(best_i)
                fb = feats[best_i]
                cb = candidates[best_i]
                for i in remaining:
                    s = cheap_similarity(candidates[i], cb, feats[i], fb, cfg.center_tau)
                    if s > max_sim[i]:
                        max_sim[i] = s

        n_slots = min(cfg.n_select, n)
        n_free = max(0, n_slots - cfg.n_elite)
        tiers = ClassTiers(cfg.freq, cfg.rare_threshold)

        # phase 1a: joint (count-bucket x class-decile) strata quotas,
        # proportional to the public prior's joint distribution
        if cfg.strata_quotas:
            by_stratum: Dict[Tuple[int, int], List[int]] = {}
            for i in range(n):
                by_stratum.setdefault(stratum_key(candidates[i], tiers), []).append(i)
            for key, frac in sorted(cfg.strata_quotas.items(), key=lambda kv: -kv[1]):
                members = sorted(by_stratum.get(key, []), key=lambda i: -v_hat[i])
                greedy(members, int(np.ceil(frac * n_free)))
        else:
            # fallback: flat minimum per count bucket
            for b in sorted(buckets):
                quota = int(np.ceil(cfg.min_bucket_frac * n_free))
                members = sorted(buckets[b], key=lambda i: -v_hat[i])
                greedy(members, min(quota, len(members)))

        # phase 1b: guaranteed survival for rare-class-dominant candidates
        chosen = set(selected)
        rare_pool = [i for i in range(n) if i not in chosen and tiers.is_rare_dominant(candidates[i])]
        n_rare_guarantee = int(np.ceil(cfg.rare_frac * n_free))
        greedy(sorted(rare_pool, key=lambda i: -v_hat[i]),
               max(0, n_rare_guarantee))

        # phase 1c: guaranteed survival per dominant-object aspect bucket
        # (wide / square / tall — a public property of the candidate that
        # top-k selection otherwise collapses onto person-like tall boxes)
        n_aspect_guarantee = int(np.ceil(0.10 * n_free))
        chosen = set(selected)
        for lo, hi in ((0.0, 0.8), (0.8, 1.25), (1.25, 1e9)):
            members = [i for i in range(n) if i not in chosen
                       and lo <= _dominant_aspect(candidates[i]) < hi]
            if members:
                members.sort(key=lambda i: -v_hat[i])
                before = len(selected)
                greedy(members, n_aspect_guarantee)
                chosen = set(selected)

        # phase 2: elite slots by pure votes
        chosen = set(selected)
        elite = [int(i) for i in np.argsort(-v_hat) if int(i) not in chosen]
        for i in elite:
            if len(selected) < n_slots:
                selected.append(i)
            else:
                break

        # phase 3: remaining slots by diversity-aware greedy over everyone
        chosen = set(selected)
        rest = [i for i in range(n) if i not in chosen]
        greedy(rest, max(0, n_slots - len(selected)))

        return selected[:n_slots]
