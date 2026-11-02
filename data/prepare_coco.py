"""Prepare private layout JSONs from COCO 2017 annotations.

Reads instances_val2017.json + captions_val2017.json, filters each image
to 2-6 "clean" objects (drops crowd boxes and tiny boxes), normalizes
bboxes to xywh in [0, 1], and writes one JSON per image plus a combined
JSONL, a class vocabulary and dataset statistics.

One image = one privacy record: all objects of an image stay together.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

# --------------------------------------------------------------- filters
MIN_REL_AREA = 0.005      # drop boxes covering < 0.5% of the image
MAX_OBJECTS = 6           # keep images with 2..6 clean objects
MIN_OBJECTS = 2
MAX_IMAGES = 5000         # cap (val2017 has exactly 5000)
RARE_CLASS_MIN = 20       # drop objects of classes rarer than this in the kept subset


def load_coco(root: str) -> tuple:
    with open(os.path.join(root, "instances_val2017.json"), "r", encoding="utf-8") as f:
        inst = json.load(f)
    cap_path = os.path.join(root, "captions_val2017.json")
    caps: Dict[int, List[str]] = defaultdict(list)
    if os.path.exists(cap_path):
        with open(cap_path, "r", encoding="utf-8") as f:
            cap = json.load(f)
        for a in cap["annotations"]:
            caps[a["image_id"]].append(a["caption"])
    return inst, caps


def prepare(coco_root: str, out_dir: str, seed: int = 0,
            max_images: int = MAX_IMAGES) -> None:
    os.makedirs(out_dir, exist_ok=True)
    inst, caps = load_coco(coco_root)
    cat_names = {c["id"]: c["name"] for c in inst["categories"]}

    imgs = {im["id"]: (im["width"], im["height"]) for im in inst["images"]}
    anns_by_img: Dict[int, list] = defaultdict(list)
    for a in inst["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns_by_img[a["image_id"]].append(a)

    rng = random.Random(seed)
    records = []
    dropped_reasons = Counter()

    for img_id, (W, H) in imgs.items():
        objs = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]  # COCO bbox is xywh absolute
            area = (w * h) / float(W * H)
            if area < MIN_REL_AREA:
                dropped_reasons["tiny_box"] += 1
                continue
            objs.append({
                "class": cat_names[a["category_id"]],
                "box": [x / W, y / H, w / W, h / H],
            })
        if not (MIN_OBJECTS <= len(objs) <= MAX_OBJECTS):
            dropped_reasons["object_count"] += 1
            continue
        first_cap = caps.get(img_id, [""])[0]
        records.append({"image_id": img_id, "caption": first_cap, "objects": objs})

    # drop rare classes (objects, not images), then re-filter counts
    cls_count = Counter(o["class"] for r in records for o in r["objects"])
    rare = {c for c, n in cls_count.items() if n < RARE_CLASS_MIN}
    if rare:
        for r in records:
            r["objects"] = [o for o in r["objects"] if o["class"] not in rare]
        records = [r for r in records if MIN_OBJECTS <= len(r["objects"]) <= MAX_OBJECTS]

    rng.shuffle(records)
    records = records[:max_images]

    # ------------------------------------------------------------- write
    per_img_dir = os.path.join(out_dir, "layouts")
    os.makedirs(per_img_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "coco_layouts.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(os.path.join(per_img_dir, f"{r['image_id']}.json"), "w", encoding="utf-8") as g:
                json.dump(r, g, ensure_ascii=False, indent=2)

    vocab = sorted({o["class"] for r in records for o in r["objects"]})
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------ stats
    n_obj = sum(len(r["objects"]) for r in records)
    counts = Counter(len(r["objects"]) for r in records)
    cls_hist = Counter(o["class"] for r in records for o in r["objects"])
    stats = {
        "n_images": len(records),
        "n_objects": n_obj,
        "objects_per_image": {str(k): v for k, v in sorted(counts.items())},
        "n_classes": len(vocab),
        "class_histogram": dict(cls_hist.most_common()),
        "dropped": dict(dropped_reasons),
        "filters": {"min_rel_area": MIN_REL_AREA, "min_objects": MIN_OBJECTS,
                    "max_objects": MAX_OBJECTS, "rare_class_min": RARE_CLASS_MIN},
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: stats[k] for k in ("n_images", "n_objects", "n_classes")},
                     ensure_ascii=False))
    print(f"objects/image: {stats['objects_per_image']}")
    print(f"wrote {jsonl_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True,
                    help="dir containing instances_val2017.json / captions_val2017.json")
    ap.add_argument("--out-dir", default="data/coco_private")
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    prepare(args.coco_root, args.out_dir, seed=args.seed, max_images=args.max_images)
