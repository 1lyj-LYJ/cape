"""Build the public layout pool (PublicPool) from a COCO-style annotation file.

The public pool may come from any *public* source: COCO train2017 (as a
stand-in for the smoke test), Visual Genome, etc. It only feeds mutation
operators and candidate seeding — never the private voting step.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List

from layout.mutation import PublicPool


def pool_from_coco(instances_json: str, classes: List[str], max_sizes_per_class: int = 50,
                   min_rel_area: float = 0.005, seed: int = 0) -> PublicPool:
    """Extract per-class size statistics + frequencies from a public COCO split.

    Streams the annotations array with ijson — instances_train2017.json is
    ~450MB and does not fit comfortably in json.load.
    """
    import ijson

    want = set(classes)
    cat_names = {}
    sizes: Dict[str, List] = defaultdict(list)
    freq: Dict[str, int] = defaultdict(int)
    rng = random.Random(seed)

    with open(instances_json, "rb") as f:
        # categories first (they appear before annotations in COCO files)
        for cat in ijson.items(f, "categories.item"):
            cat_names[cat["id"]] = cat["name"]

    with open(instances_json, "rb") as f:
        # images map for width/height
        imgs = {}
        for im in ijson.items(f, "images.item"):
            imgs[im["id"]] = (im["width"], im["height"])

    with open(instances_json, "rb") as f:
        for a in ijson.items(f, "annotations.item"):
            if a.get("iscrowd", 0):
                continue
            cls = cat_names.get(a["category_id"])
            if cls not in want:
                continue
            W, H = imgs[a["image_id"]]
            x, y, w, h = (float(v) for v in a["bbox"])
            W, H = float(W), float(H)
            if (w * h) / float(W * H) < min_rel_area:
                continue
            freq[cls] += 1
            # reservoir-style random sample so annotation order cannot
            # bias the size statistics
            if len(sizes[cls]) < max_sizes_per_class:
                sizes[cls].append((max(w / W, 0.02), max(h / H, 0.02)))
            elif rng.random() < max_sizes_per_class / (freq[cls] + 1):
                sizes[cls][rng.randrange(max_sizes_per_class)] = (
                    max(w / W, 0.02), max(h / H, 0.02))

    # object-count distribution of public images (for seeding k)
    with open(instances_json, "rb") as f:
        per_img: Dict[int, int] = defaultdict(int)
        for a in ijson.items(f, "annotations.item"):
            if a.get("iscrowd", 0):
                continue
            if cat_names.get(a["category_id"]) in want:
                per_img[a["image_id"]] += 1
        counts = [n for n in per_img.values() if 1 <= n <= 8]

    return PublicPool(
        classes=classes,
        sizes_by_class={c: v for c, v in sizes.items() if v},
        class_freq=dict(freq),
        counts=counts,
    )
