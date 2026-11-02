"""Parallel vote histogram for the private-evolution query.

The round-t query is embarrassingly parallel over PRIVATE records: each private
layout independently finds its nearest candidate and casts one vote.  This module
therefore parallelises over the private set only, and is **numerically
identical** to the serial path in ``pe.engine._vote_matrix``:

  * votes are integer counts (addition commutes);
  * ``argmin`` ties are broken by the LOWEST candidate index -- reproduced here
    with a strict ``<`` comparison, which keeps the first minimum;
  * the returned nn-distance list preserves the original private-record order,
    so ``np.mean`` over it matches the serial value bit for bit.

Relation term.  When ``dist_cfg.classes`` is set the composite distance includes
``eta * d_rel``.  ``layout_distance`` would recompute two dense (C,R,C) =
80*7*80 = 44,800-float relation histograms *per pair* (N*K = 2.86e6 pairs per
round -> ~1 TB of allocation traffic).  Instead we hoist the relation histogram
to a layout-level feature (``layout.relation_cache``), extracted once per layout
(N+K per round instead of 2*N*K), and add ``eta * rel_tv(phi_a, phi_b)`` to the
relation-free distance.  This is the same value: verified in
``scripts/query_rel_gate.py`` (max|d1_cached - d1_naive| = 1.8e-15) and in
``scripts/verify_vote_rel.py`` at the vote level.

Displacement term.  When ``dist_cfg.zeta > 0`` the composite distance gains
``zeta * d_disp`` (pairwise relative-displacement disagreement over matched
pairs, see ``layout.distance._displacement_field``).  Computed inline in the
worker loop from the padded cost-matrix path; the matched pairs come from the
MatchResult.

Default ``n_jobs=1`` runs in-process and changes nothing.
"""
from __future__ import annotations

import dataclasses
from typing import List, Sequence, Tuple

import numpy as np

from layout.distance import DistanceConfig, layout_distance
from layout.relation_cache import phi_rel, rel_tv
from layout.representation import Layout

_W = {}


def _init_worker(candidates, base_cfg, phi_c, eta, classes):
    import os

    from layout.matching import MatchingConfig

    _W["cands"] = candidates
    _W["cfg"] = base_cfg          # relation-free config (classes=None)
    _W["phic"] = phi_c            # cached candidate relation features, or None
    _W["eta"] = eta
    _W["classes"] = classes       # needed inside the worker (spawn = fresh state)
    # PV_DISTANCE=emd|emd_cnt switches the assignment rule only (OT baseline);
    # spawn workers inherit the environment, so no initargs change is needed.
    mode = os.environ.get("PV_DISTANCE", "hungarian").strip().lower()
    _W["mode"] = mode
    _W["lam_cnt"] = float(os.environ.get("PV_LAM_CNT", "2.5"))
    if mode in ("emd", "emd_cnt"):
        _W["emd_cfg"] = MatchingConfig()


def _vote_chunk(privs: Sequence[Layout]) -> List[Tuple[int, float]]:
    cands = _W["cands"]
    cfg = _W["cfg"]
    phic = _W["phic"]
    eta = _W["eta"]
    classes = _W["classes"]
    mode = _W.get("mode", "hungarian")
    lam_cnt = _W.get("lam_cnt", 0.0)
    out: List[Tuple[int, float]] = []
    if mode == "ltsim":
        from layout.ltsim import ltsim_distance
        for priv in privs:
            best_j, best_d = 0, float("inf")
            for j, c in enumerate(cands):
                d = ltsim_distance(priv, c)
                if d < best_d:              # strict '<' keeps the FIRST minimum
                    best_d, best_j = d, j
            out.append((best_j, best_d))
        return out
    if mode == "ltsim_cnt":
        # LTSim OT cost + the SAME cardinality penalty as d_CA's mu term
        from layout.ltsim import ltsim_distance
        for priv in privs:
            m = max(len(priv), 1)
            best_j, best_d = 0, float("inf")
            for j, c in enumerate(cands):
                d = ltsim_distance(priv, c) + lam_cnt * abs(len(priv) - len(c)) / m
                if d < best_d:
                    best_d, best_j = d, j
            out.append((best_j, best_d))
        return out
    if mode == "chamfer_cnt":
        # Chamfer pipeline + mu-analog term; PV_GAMMA controls the native
        # gamma count term inside layout_distance (default 0.5 = legacy;
        # 0.0 = single-penalty fair grid matching the simplified CA-PE).
        from layout.distance import DistanceConfig as _DC
        from layout.matching import MatchingConfig as _MC
        import os as _os
        cfg_c = _DC(matching=_MC(method="chamfer"),
                    gamma=float(_os.environ.get("PV_GAMMA", "0.5")))
        for priv in privs:
            m = max(len(priv), 1)
            best_j, best_d = 0, float("inf")
            for j, c in enumerate(cands):
                d = layout_distance(priv, c, cfg_c)["total"] \
                    + lam_cnt * abs(len(priv) - len(c)) / m
                if d < best_d:
                    best_d, best_j = d, j
            out.append((best_j, best_d))
        return out
    if mode in ("emd", "emd_cnt"):
        from layout.emd import emd_layout
        emd_cfg = _W["emd_cfg"]
        for priv in privs:
            best_j, best_d = 0, float("inf")
            for j, c in enumerate(cands):
                d = emd_layout(priv, c, emd_cfg,
                               lam_cnt=(lam_cnt if mode == "emd_cnt" else 0.0))
                if d < best_d:              # strict '<' keeps the FIRST minimum
                    best_d, best_j = d, j
            out.append((best_j, best_d))
        return out
    for priv in privs:
        phi_p = phi_rel(priv, classes) if phic is not None else None
        best_j, best_d = 0, float("inf")
        for j, c in enumerate(cands):
            r = layout_distance(priv, c, cfg)
            d = r["total"]
            if phic is not None:
                d += eta * rel_tv(phi_p, phic[j])
            if d < best_d:              # strict '<' keeps the FIRST minimum,
                best_d, best_j = d, j   # identical to np.argmin tie-breaking
        out.append((best_j, best_d))
    return out


def vote_histogram(private: Sequence[Layout], candidates: Sequence[Layout],
                   dist_cfg: DistanceConfig, n_jobs: int = 1
                   ) -> Tuple[np.ndarray, float]:
    """Return (votes over candidates, mean nearest-candidate distance)."""
    n_priv = len(private)
    votes = np.zeros(len(candidates), dtype=np.float64)
    if n_priv == 0:
        return votes, 0.0

    # vectorized flattened-distance modes (baselines through the same engine):
    # PV_DISTANCE=flat | sortedflat  -> slot-aligned Euclidean distance on
    # one-hot+box slot vectors, identical to the Tab-PE-style flat protocol
    import os
    pv_mode = os.environ.get("PV_DISTANCE", "hungarian").strip().lower()
    if pv_mode in ("flat", "sortedflat", "sortedflat_cnt"):
        import json
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab = json.load(open(os.path.join(root, "configs",
                                            "coco80_classes.json"),
                               encoding="utf-8"))
        c2i = {c: i for i, c in enumerate(vocab)}
        n_cls, max_obj = len(vocab), 6

        def _flat(lay):
            v = np.zeros(max_obj * (n_cls + 4), dtype=np.float64)
            objs = (sorted(lay.objects,
                           key=lambda o: (c2i.get(o.cls, n_cls), o.box.x,
                                          o.box.y, o.box.w, o.box.h))
                    if pv_mode in ("sortedflat", "sortedflat_cnt")
                    else lay.objects)
            for i, o in enumerate(objs[:max_obj]):
                base = i * (n_cls + 4)
                v[base + c2i.get(o.cls, n_cls)] = 1.0
                v[base + n_cls:base + n_cls + 4] = (o.box.x, o.box.y,
                                                    o.box.w, o.box.h)
            return v

        P = np.stack([_flat(l) for l in private])
        C = np.stack([_flat(c) for c in candidates])
        d2 = ((P * P).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2 * P @ C.T)
        D = np.sqrt(np.maximum(d2, 0.0)) / max_obj
        if pv_mode == "sortedflat_cnt":
            # add the same cardinality term as the paper's d_CA count control
            mi = np.array([len(l.objects) for l in private],
                          dtype=np.float64)[:, None]
            nk = np.array([len(c.objects) for c in candidates],
                          dtype=np.float64)[None, :]
            D = D + float(os.environ.get("PV_LAM_CNT", "2.5")) * (
                np.abs(mi - nk) / np.maximum(mi, nk))
        nn_idx = D.argmin(1)
        for j in nn_idx:
            votes[j] += 1.0
        return votes, float(np.mean(D[np.arange(n_priv), nn_idx]))

    use_rel = bool(dist_cfg.eta > 0 and dist_cfg.classes is not None)
    if use_rel:
        base_cfg = dataclasses.replace(dist_cfg, classes=None)
        phi_c = [phi_rel(c, dist_cfg.classes) for c in candidates]
        eta, classes = dist_cfg.eta, list(dist_cfg.classes)
    else:
        base_cfg, phi_c, eta, classes = dist_cfg, None, None, None

    if n_jobs is None or n_jobs <= 1:
        _init_worker(list(candidates), base_cfg, phi_c, eta, classes)
        res = _vote_chunk(list(private))
    else:
        import multiprocessing as mp
        n_chunks = max(n_jobs, min(8 * n_jobs, n_priv))
        size = (n_priv + n_chunks - 1) // n_chunks
        chunks = [list(private[i:i + size]) for i in range(0, n_priv, size)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs, initializer=_init_worker,
                      initargs=(list(candidates), base_cfg, phi_c, eta, classes)) as pool:
            parts = pool.map(_vote_chunk, chunks)
        res = [r for part in parts for r in part]   # original order preserved

    nn = []
    for j, d in res:
        votes[j] += 1.0
        nn.append(d)
    return votes, float(np.mean(nn))
