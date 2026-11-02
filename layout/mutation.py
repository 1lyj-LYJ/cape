"""Relation-aware mutations over candidate layouts.

All operators are public (they may use statistics of the *public* layout
pool but never private data). They deliberately preserve or manipulate
relations between objects instead of perturbing every bbox independently.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .projection import ProjectionConfig, project_layout
from .relations import pair_relation
from .representation import Box, Layout, Object


@dataclass
class MutationConfig:
    # geometry
    translate_sigma: float = 0.05
    scale_range: Tuple[float, float] = (0.8, 1.25)
    aspect_jitter: float = 0.1
    # relations / groups
    group_sigma: float = 0.05          # cluster translation magnitude
    group_link_dist: float = 0.35      # objects whose boxes are closer than this co-move
    # add / delete
    add_delete_prob: float = 0.35
    max_objects: int = 12
    min_objects: int = 1
    # class swap
    class_swap_prob: float = 0.15

    rng: random.Random = field(default_factory=random.Random)


@dataclass
class PublicPool:
    """Public statistics used by additive mutations (Visual Genome etc.)."""
    classes: List[str]
    # per-class lists of (w, h) normalized sizes sampled from public layouts
    sizes_by_class: dict = field(default_factory=dict)
    class_freq: Optional[dict] = None  # optional sampling weights
    counts: Optional[List[int]] = None  # public per-image object counts

    def sample_class(self, rng: random.Random) -> str:
        if self.class_freq:
            classes = list(self.class_freq.keys())
            weights = np.array([self.class_freq[c] for c in classes], dtype=np.float64)
            weights /= weights.sum()
            return str(rng.choices(classes, weights=weights.tolist(), k=1)[0])
        return rng.choice(self.classes)

    def sample_count(self, rng: random.Random, lo: int = 2, hi: int = 6) -> int:
        if self.counts:
            in_range = [c for c in self.counts if lo <= c <= hi]
            if in_range:
                return in_range[rng.randrange(len(in_range))]
        return rng.randint(lo, hi)

    def sample_box(self, cls: str, rng: random.Random) -> Box:
        sizes = self.sizes_by_class.get(cls)
        if sizes:
            w, h = sizes[rng.randrange(len(sizes))]
            # jitter size a little
            w = float(np.clip(w * rng.uniform(0.85, 1.15), 0.02, 0.98))
            h = float(np.clip(h * rng.uniform(0.85, 1.15), 0.02, 0.98))
        else:
            w = rng.uniform(0.1, 0.5)
            h = rng.uniform(0.1, 0.5)
        x = rng.uniform(0.0, max(0.0, 1.0 - w))
        y = rng.uniform(0.0, max(0.0, 1.0 - h))
        return Box(x, y, w, h)


def _translate(b: Box, dx: float, dy: float) -> Box:
    return Box(b.x + dx, b.y + dy, b.w, b.h)


def _scale(b: Box, s: float, rng: random.Random) -> Box:
    w, h = b.w * s, b.h * s
    cx, cy = b.cx, b.cy
    return Box(cx - w / 2.0, cy - h / 2.0, w, h)


def _aspect_jitter(b: Box, eps: float, rng: random.Random) -> Box:
    log_ar = np.log(b.w / max(b.h, 1e-9))
    log_ar += rng.uniform(-eps, eps)
    h = np.sqrt(b.area / np.exp(log_ar))
    w = np.exp(log_ar) * h
    cx, cy = b.cx, b.cy
    return Box(cx - w / 2.0, cy - h / 2.0, w, h)


def _clusters(layout: Layout, link_dist: float) -> List[List[int]]:
    """Union-find grouping of objects whose boxes are close (IoU>0 or center
    distance < link_dist). Used for relation-preserving group moves."""
    n = len(layout.objects)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = layout.objects[i].box, layout.objects[j].box
            ci = np.array([bi.cx, bi.cy])
            cj = np.array([bj.cx, bj.cy])
            if bi.iou(bj) > 0 or np.linalg.norm(ci - cj) < link_dist:
                union(i, j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def mutate(layout: Layout, pool: PublicPool, cfg: Optional[MutationConfig] = None,
           proj_cfg: Optional[ProjectionConfig] = None,
           mode: str = "full") -> Layout:
    """Apply one mutation to a copy of ``layout``.

    mode="full":    all operators (geometry / add / delete / relation flip /
                    class swap / jitter) — for low-fitness survivors that
                    should explore aggressively.
    mode="refine":  geometry-only with half magnitudes — for high-fitness
                    parents whose gross structure is already good.
    """
    cfg = cfg or MutationConfig()
    rng = cfg.rng
    if not layout.objects:
        # empty candidate: seed with 1-3 objects from the public pool
        objs = [Object(cls=pool.sample_class(rng), box=pool.sample_box(pool.sample_class(rng), rng))
                for _ in range(rng.randint(1, 3))]
        return project_layout(Layout(objects=objs), proj_cfg)

    objs = [Object(cls=o.cls, box=Box(o.box.x, o.box.y, o.box.w, o.box.h))
            for o in layout.objects]
    L = Layout(objects=objs)

    if mode == "refine":
        i = rng.randrange(len(L.objects))
        op = rng.random()
        b = L.objects[i].box
        if op < 0.35:
            groups = _clusters(L, cfg.group_link_dist)
            g = groups[rng.randrange(len(groups))]
            dx = rng.gauss(0, cfg.group_sigma / 2)
            dy = rng.gauss(0, cfg.group_sigma / 2)
            for j in g:
                L.objects[j].box = _translate(L.objects[j].box, dx, dy)
        elif op < 0.65:
            L.objects[i].box = _translate(b, rng.gauss(0, cfg.translate_sigma / 2),
                                          rng.gauss(0, cfg.translate_sigma / 2))
        elif op < 0.9:
            L.objects[i].box = _scale(b, rng.uniform(0.9, 1.1), rng)
        else:
            L.objects[i].box = _aspect_jitter(b, cfg.aspect_jitter / 2, rng)
        return project_layout(L, proj_cfg)

    choice = rng.random()
    # count random-walk is the main diversity killer: add/delete stay rare
    # exploratory moves; the common cardinality op is REPLACE (delete+add,
    # count-preserving) so object counts drift only slowly
    p_add, p_del, p_repl = 0.03, 0.03, 0.20
    op = None

    if choice < 0.15:
        # group translation: move a cluster of related objects together,
        # preserving their relative geometry
        groups = _clusters(L, cfg.group_link_dist)
        g = groups[rng.randrange(len(groups))]
        dx = rng.gauss(0, cfg.group_sigma)
        dy = rng.gauss(0, cfg.group_sigma)
        op = "group"
        for i in g:
            L.objects[i].box = _translate(L.objects[i].box, dx, dy)

    elif choice < 0.40:
        # single-object geometry mutation
        i = rng.randrange(len(L.objects))
        op = rng.random()
        b = L.objects[i].box
        if op < 0.4:
            b = _translate(b, rng.gauss(0, cfg.translate_sigma), rng.gauss(0, cfg.translate_sigma))
        elif op < 0.75:
            b = _scale(b, rng.uniform(*cfg.scale_range), rng)
        else:
            b = _aspect_jitter(b, cfg.aspect_jitter, rng)
        op = "geometry"
        L.objects[i].box = b

    elif choice < 0.40 + p_add and len(L.objects) < cfg.max_objects:
        # add: place a new object *near* an existing one of a compatible
        # relation (near / right_of), so additions respect scene structure
        anchor = L.objects[rng.randrange(len(L.objects))]
        cls = pool.sample_class(rng)
        b = pool.sample_box(cls, rng)
        # pull the new box toward the anchor so it lands in a plausible relation
        dx = (anchor.box.cx - b.cx) * 0.6 + rng.gauss(0, cfg.translate_sigma)
        dy = (anchor.box.cy - b.cy) * 0.6 + rng.gauss(0, cfg.translate_sigma)
        b = _translate(b, dx, dy)
        op = "add"
        L.objects.append(Object(cls=cls, box=b))

    elif choice < 0.40 + p_add + p_del and len(L.objects) > cfg.min_objects:
        # delete a random object
        op = "delete"
        L.objects.pop(rng.randrange(len(L.objects)))

    elif choice < 0.40 + p_add + p_del + p_repl and len(L.objects) > cfg.min_objects:
        # replace: delete one object, add another — count-preserving
        op = "replace"
        L.objects.pop(rng.randrange(len(L.objects)))
        anchor = L.objects[rng.randrange(len(L.objects))]
        cls = pool.sample_class(rng)
        b = pool.sample_box(cls, rng)
        dx = (anchor.box.cx - b.cx) * 0.6 + rng.gauss(0, cfg.translate_sigma)
        dy = (anchor.box.cy - b.cy) * 0.6 + rng.gauss(0, cfg.translate_sigma)
        L.objects.append(Object(cls=cls, box=_translate(b, dx, dy)))

    elif choice < 0.75 + p_repl and len(L.objects) >= 2:
        # relation mutation: mirror one object across the other along the
        # dominant axis, flipping the directional relation (right_of ->
        # left_of, ...) while each object keeps its class and size
        i = rng.randrange(len(L.objects))
        j = (i + 1 + rng.randrange(len(L.objects) - 1)) % len(L.objects) if len(L.objects) > 1 else i
        if i != j:
            bi, bj = L.objects[i].box, L.objects[j].box
            if pair_relation(bi, bj) in ("left_of", "right_of", "above", "below"):
                op = "flip"
                dx = bj.cx - bi.cx
                dy = bj.cy - bi.cy
                if abs(dx) >= abs(dy):
                    ncx = bi.cx - dx  # mirror across the anchor's center
                    L.objects[j].box = _translate(bj, ncx - bj.cx, 0.0)
                else:
                    ncy = bi.cy - dy
                    L.objects[j].box = _translate(bj, 0.0, ncy - bj.cy)

    elif choice < 0.85 + p_repl and pool.classes:
        # class swap of a single object (keeps geometry, changes semantics)
        i = rng.randrange(len(L.objects))
        op = "class_swap"
        if pool.class_freq:
            # frequency-weighted over classes != current
            others = [(c, w) for c, w in pool.class_freq.items() if c != L.objects[i].cls]
            if others:
                cs, ws = zip(*others)
                L.objects[i].cls = str(rng.choices(cs, weights=ws, k=1)[0])
            else:
                L.objects[i].cls = pool.sample_class(rng)
        else:
            others = [c for c in pool.classes if c != L.objects[i].cls] or pool.classes
            L.objects[i].cls = others[rng.randrange(len(others))]

    else:
        # global jitter: small independent nudges for fine-grained exploration
        op = "jitter"
        for o in L.objects:
            o.box = _translate(o.box, rng.gauss(0, cfg.translate_sigma / 2),
                               rng.gauss(0, cfg.translate_sigma / 2))

    cfg.last_op = op
    return project_layout(L, proj_cfg)


def mutate_n(layout: Layout, pool: PublicPool, steps: int,
             cfg: Optional[MutationConfig] = None,
             proj_cfg: Optional[ProjectionConfig] = None,
             mode: str = "full") -> Layout:
    """Apply ``steps`` successive mutations (compound exploration move)."""
    L = layout
    for _ in range(steps):
        L = mutate(L, pool, cfg, proj_cfg, mode=mode)
    return L


def seed_candidates(pool: PublicPool, n: int, cfg: Optional[MutationConfig] = None) -> List[Layout]:
    """Initial candidate population drawn purely from public statistics."""
    cfg = cfg or MutationConfig()
    rng = cfg.rng
    out = []
    for _ in range(n):
        k = pool.sample_count(rng)
        objs = []
        for _ in range(k):
            cls = pool.sample_class(rng)
            objs.append(Object(cls=cls, box=pool.sample_box(cls, rng)))
        out.append(project_layout(Layout(objects=objs)))
    return out
