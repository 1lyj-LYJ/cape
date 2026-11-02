"""Layout representation: an unordered, variable-cardinality set of objects.

A layout is a set (never a vector) of objects ``o_i = (class, x, y, w, h)``
with boxes normalized to [0, 1] in xywh format (top-left corner + size).
All downstream code must be permutation-invariant with respect to the
object list order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Box:
    """Normalized bbox in xywh (corner + size). All values in [0, 1]."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> "Box":
        return Box(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))

    def clipped(self) -> "Box":
        """Project onto the unit square, keeping the box non-degenerate."""
        x1, y1, x2, y2 = self.to_xyxy()
        x1, y1 = min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0)
        x2, y2 = min(max(x2, 0.0), 1.0), min(max(y2, 0.0), 1.0)
        return Box.from_xyxy(x1, y1, x2, y2)

    def iou(self, other: "Box") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def giou(self, other: "Box") -> float:
        """Generalized IoU in [-1, 1] (Rezatofighi et al., CVPR 2019)."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        iou = inter / union if union > 0 else 0.0
        # smallest enclosing box
        ex1, ey1 = min(self.x, other.x), min(self.y, other.y)
        ex2 = max(self.x + self.w, other.x + other.w)
        ey2 = max(self.y + self.h, other.y + other.h)
        enclose = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
        return iou - (enclose - union) / enclose if enclose > 0 else iou


@dataclass
class Object:
    cls: str
    box: Box


@dataclass
class Layout:
    objects: List[Object] = field(default_factory=list)
    image_id: int = -1
    caption: str = ""

    def __len__(self) -> int:
        return len(self.objects)

    def classes(self) -> List[str]:
        return [o.cls for o in self.objects]

    def boxes(self) -> np.ndarray:
        """(N, 4) array of xywh boxes."""
        if not self.objects:
            return np.zeros((0, 4), dtype=np.float64)
        return np.array([[o.box.x, o.box.y, o.box.w, o.box.h] for o in self.objects])

    def sorted(self) -> "Layout":
        """Canonical ordering (class, then box) — for hashing/dedup only.

        Distance and metrics must never rely on this; it exists so that
        two identical layouts compare equal in dicts/tests.
        """
        objs = sorted(self.objects, key=lambda o: (o.cls, o.box.x, o.box.y, o.box.w, o.box.h))
        return Layout(objects=objs, image_id=self.image_id, caption=self.caption)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "caption": self.caption,
            "objects": [
                {"class": o.cls, "box": [round(o.box.x, 4), round(o.box.y, 4),
                                         round(o.box.w, 4), round(o.box.h, 4)]}
                for o in self.objects
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        return Layout(
            image_id=d.get("image_id", -1),
            caption=d.get("caption", ""),
            objects=[Object(cls=o["class"], box=Box(*o["box"])) for o in d["objects"]],
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Layout":
        with open(path, "r", encoding="utf-8") as f:
            return Layout.from_dict(json.load(f))


def gligen_boxes(layout: Layout) -> Tuple[List[str], List[List[float]]]:
    """Convert a layout into GLIGEN's (phrases, locations) interface.

    GLIGEN expects normalized xyxy boxes [x1, y1, x2, y2] with x1 < x2.
    """
    phrases = [o.cls for o in layout.objects]
    locations = []
    for o in layout.objects:
        x1, y1, x2, y2 = o.box.to_xyxy()
        x2 = max(x2, x1 + 1e-3)
        y2 = max(y2, y1 + 1e-3)
        locations.append([x1, y1, x2, y2])
    return phrases, locations
