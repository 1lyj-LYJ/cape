"""Constraint projection: map any layout back to the valid layout space.

Projection never touches private data — it is a deterministic post-
processing of the DP output and therefore costs no additional epsilon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .representation import Box, Layout, Object


@dataclass
class ProjectionConfig:
    min_size: float = 0.02        # min w / h
    min_area: float = 0.005       # min bbox area
    max_area: float = 0.95        # a single object cannot cover the canvas
    max_objects: int = 12
    max_pair_iou: float = 0.9     # near-duplicate suppression
    aspect_range: tuple = (0.15, 6.0)  # valid w/h aspect ratio range


def project_box(b: Box, cfg: ProjectionConfig) -> Optional[Box]:
    """Clamp to canvas, enforce min size/area/aspect. Returns None if unrecoverable."""
    b = b.clipped()
    if b.w < cfg.min_size:
        if b.x + cfg.min_size > 1.0:
            b.x = max(0.0, 1.0 - cfg.min_size)
        b.w = cfg.min_size
    if b.h < cfg.min_size:
        if b.y + cfg.min_size > 1.0:
            b.y = max(0.0, 1.0 - cfg.min_size)
        b.h = cfg.min_size

    # max area: shrink toward center (capture the center before resizing,
    # since Box.cx depends on the current w/h)
    if b.area > cfg.max_area:
        cx, cy = b.cx, b.cy
        s = (cfg.max_area / b.area) ** 0.5
        b.w, b.h = b.w * s, b.h * s
        b.x, b.y = cx - b.w / 2.0, cy - b.h / 2.0
        b = b.clipped()

    # aspect ratio: stretch the smaller dimension around the same center
    lo, hi = cfg.aspect_range
    ar = b.w / max(b.h, 1e-9)
    if ar < lo:
        cx, cy = b.cx, b.cy
        b.w = lo * b.h
        b.x = cx - b.w / 2.0
        b = b.clipped()
    elif ar > hi:
        cx, cy = b.cx, b.cy
        b.h = b.w / hi
        b.y = cy - b.h / 2.0
        b = b.clipped()

    # re-clamp may have shrunk below min size again — give up if degenerate
    if b.w < 0.5 * cfg.min_size or b.h < 0.5 * cfg.min_size or b.area < 0.25 * cfg.min_area:
        return None
    return b


def project_layout(layout: Layout, cfg: Optional[ProjectionConfig] = None) -> Layout:
    """Project a layout onto the valid set: boxes legal, duplicates removed,
    cardinality capped."""
    cfg = cfg or ProjectionConfig()
    out: List[Object] = []
    for obj in layout.objects:
        if len(out) >= cfg.max_objects:
            break
        pb = project_box(obj.box, cfg)
        if pb is None:
            continue
        # near-duplicate of an already-projected object (same class, IoU too high)
        dup = any(o.cls == obj.cls and o.box.iou(pb) >= cfg.max_pair_iou for o in out)
        if dup:
            continue
        out.append(Object(cls=obj.cls, box=pb))
    return Layout(objects=out, image_id=layout.image_id, caption=layout.caption)
