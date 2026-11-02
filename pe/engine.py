"""Layout-PE engine: evolutionary synthesis of layouts under DP.

Each round:
  1. every private layout votes for its nearest candidate (Hungarian
     layout distance) — one vote per image (= one privacy record);
  2. vote counts are released with Gaussian noise (DP histogram) —
     with ``sigma=None`` this is the non-private upper bound (eps=inf);
  3. top candidates become parents; relation-aware mutation + constraint
     projection generate the next generation.

Selection noise and the RDP accountant live in ``voting.py`` /
``accountant.py``; the engine takes a ``SelectionRule`` so the DP and
non-private paths share all other code.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from evaluation.layout_metrics import layout_set_report
from layout.distance import DistanceConfig, layout_distance
from layout.mutation import MutationConfig, PublicPool, mutate, mutate_n, seed_candidates
from layout.projection import ProjectionConfig
from layout.representation import Layout


@dataclass
class PEConfig:
    n_candidates: int = 400
    n_parents: int = 100          # survivors kept as elites each round
    n_children_per_parent: int = 2
    max_mutation_steps: int = 3   # children get 1..steps compound mutations
    n_random_survivors: int = 40  # diversity guard: random non-elite survivors
    n_rounds: int = 10
    n_seeds_from_public: int = 100  # fresh public samples injected each round
    refine_quantile: float = 0.25   # parents with fitness above this quantile refine
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    seed: int = 0
    # ---- DP plumbing ----
    sigma: Optional[float] = None   # Gaussian std for the per-round vote release;
                                    # None = eps=inf (non-private) mode
    dp_seed: int = 12345            # independent RNG stream for DP noise
    n_jobs: int = 1                 # >1 parallelises the vote query over the
                                    # private set; numerically identical to 1
    # PRIVACY BOUNDARY: _vote_matrix() touches private data. Its raw output
    # must ONLY flow into release_votes(). Everything downstream — selection,
    # mutation, projection, the final released population — must be a
    # function of (noisy votes, public candidates) alone.
    # The FINAL candidate population IS the released synthetic layout set;
    # there is no additional release step.


SelectionRule = Callable[[np.ndarray, List[Layout], random.Random], List[int]]
"""Given (noisy) votes, the candidate population and the engine RNG,
return the indices of the selected parents. DP-secure: any function of
the noisy histogram and public candidates is post-processing."""


def topk_selection(k: int, n_random: int = 0) -> SelectionRule:
    """Plain top-k by votes, plus n_random random survivors (baseline)."""
    def select(votes: np.ndarray, candidates: List[Layout], rng: random.Random) -> List[int]:
        order = list(np.argsort(-votes))
        elite = order[:k]
        if n_random > 0 and len(order) > k:
            rest = order[k:]
            elite += [rest[i] for i in rng.sample(range(len(rest)), min(n_random, len(rest)))]
        return elite
    return select


@dataclass
class RoundLog:
    round: int
    best_votes: float
    mean_private_nn_dist: float   # mean distance from private layouts to candidates
    n_unique_candidates: int
    metrics: Dict[str, float]
    wall_seconds: float
    top_candidate: Optional[dict] = None  # digest of the top-voted candidate
                                          # (by NOISY votes — DP-safe diagnostic)


class LayoutPE:
    def __init__(self, private: List[Layout], pool: PublicPool, cfg: PEConfig,
                 selection: SelectionRule):
        self.private = private
        self.pool = pool
        self.cfg = cfg
        self.selection = selection
        self.classes = pool.classes
        rng = random.Random(cfg.seed)
        self.rng = rng
        self._round = -1
        # initial candidate population: purely public samples + mutations
        self.candidates: List[Layout] = seed_candidates(pool, cfg.n_candidates, cfg.mutation)

    # ------------------------------------------------------------------
    def _vote_matrix(self) -> np.ndarray:
        """votes[i, j] = 1 if candidate j is the nearest to private layout i."""
        from pe.parallel_vote import vote_histogram
        return vote_histogram(self.private, self.candidates,
                              self.cfg.distance, self.cfg.n_jobs)

    def _set_metrics(self) -> Dict[str, float]:
        # subsample private set for speed in the smoke loop
        priv = self.private if len(self.private) <= 800 else random.Random(0).sample(self.private, 800)
        return layout_set_report(priv, self.candidates, self.classes)

    # ------------------------------------------------------------------
    def run(self, log_path: Optional[str] = None) -> List[RoundLog]:
        logs: List[RoundLog] = []
        for self._round in range(self.cfg.n_rounds):
            t0 = time.time()
            true_votes, nn_dist = self._vote_matrix()   # <-- private data enters here
            noisy = self.release_votes(true_votes)      # <-- and leaves here, once
            parent_idx = self.selection(noisy, self.candidates, self.rng)

            # fitness-aware mutation strength: high-vote parents refine,
            # low-vote (e.g. bucket-guaranteed) survivors explore
            v = np.clip(noisy, 0.0, None)
            vmax = float(v.max()) if len(v) else 0.0
            parent_fit = [float(v[i]) / vmax if vmax > 0 else 0.0 for i in parent_idx]
            thresh = float(np.quantile(parent_fit, self.cfg.refine_quantile)) if parent_fit else 0.0

            # next generation: parents' children + fresh public seeds
            children: List[Layout] = []
            for i, p_idx in enumerate(parent_idx):
                p = self.candidates[p_idx]
                if parent_fit[i] >= thresh:
                    # high fitness: geometry-only refinement, one step
                    mode, steps = "refine", 1
                else:
                    mode = "full"
                    steps = self.rng.randint(1, max(1, self.cfg.max_mutation_steps))
                for _ in range(self.cfg.n_children_per_parent):
                    children.append(mutate_n(p, self.pool, steps, self.cfg.mutation,
                                             self.cfg.projection, mode=mode))
            fresh = []
            if self.cfg.n_seeds_from_public > 0:
                fresh = seed_candidates(self.pool, self.cfg.n_seeds_from_public, self.cfg.mutation)
            # the selected parents themselves survive as elites
            self.candidates = [self.candidates[i] for i in parent_idx] + children + fresh

            log = RoundLog(
                round=self._round,
                best_votes=float(noisy.max()) if len(noisy) else 0.0,
                mean_private_nn_dist=nn_dist,
                n_unique_candidates=len({tuple(sorted((o.cls, o.box.x, o.box.y, o.box.w, o.box.h)
                                                      for o in c.objects)) for c in self.candidates}),
                metrics=self._set_metrics(),
                wall_seconds=time.time() - t0,
                top_candidate=(self.candidates[int(np.argmax(noisy))].to_dict()
                               if len(noisy) else None),
            )
            logs.append(log)
            print(f"[round {self._round}] nn_dist={nn_dist:.3f} best_votes={log.best_votes:.0f} "
                  f"count_TV={log.metrics['count_TV']:.3f} rel_TV={log.metrics['relation_TV']:.3f} "
                  f"class_TV={log.metrics['class_TV']:.3f} ({log.wall_seconds:.1f}s)")
            if log_path:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**log.__dict__, "round": self._round}, default=float) + "\n")
        return logs

    def synthetic_nearest(self) -> List[Layout]:
        """Diagnostic (eps=inf only): each private layout is replaced by its
        nearest candidate. Uses private data directly — never a DP release."""
        out = []
        for priv in self.private:
            dists = [layout_distance(priv, c, self.cfg.distance)["total"]
                     for c in self.candidates]
            out.append(self.candidates[int(np.argmin(dists))])
        return out

    def release_votes(self, true_votes: np.ndarray) -> np.ndarray:
        """The single privacy boundary crossing.

        eps=inf mode (sigma=None): identity. DP mode: true_votes +
        Gaussian noise with std sigma (add/remove image-level adjacency,
        L2 sensitivity 1), drawn from an RNG stream independent of the
        selection/mutation RNG. Everything downstream of this function
        must consume ONLY the returned array.
        """
        if self.cfg.sigma is None:
            return true_votes.copy()
        noise_rng = np.random.default_rng(self.cfg.dp_seed + 1000 * (self._round + 1))
        return true_votes + noise_rng.normal(0.0, self.cfg.sigma, size=true_votes.shape)

    def synthetic_population(self) -> List[Layout]:
        """THE release: the final candidate population IS the DP synthetic
        layout dataset (post-processing of the T DP vote releases plus the
        public pool — no private data involved). Expand to more layouts by
        uniform resampling + public mutation, still post-processing."""
        return list(self.candidates)

    # ------------------------------------------------------------------
    # The two methods below are eps=inf DIAGNOSTICS ONLY: both touch
    # private data (raw votes / private layouts) and must never be part
    # of the DP release path.
    # ------------------------------------------------------------------
    def synthetic_from_votes(self, n: int, noisy_votes: Optional[np.ndarray] = None) -> List[Layout]:
        """Diagnostic: sample candidates proportional to (noisy) votes."""
        if noisy_votes is None:
            votes, _ = self._vote_matrix()
            noisy_votes = self.release_votes(votes)
        w = np.clip(noisy_votes, 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        idx = self.rng.choices(range(len(self.candidates)), weights=w.tolist(), k=n)
        return [self.candidates[i] for i in idx]

    def synthetic_layouts(self, n: int) -> List[Layout]:
        """Deprecated alias for synthetic_from_votes."""
        return self.synthetic_from_votes(n)
