"""Prepare the VG150 second dataset (COCO-format parquet with embedded images).

Outputs (mirrors data/coco_private structure):
  data/vg150_private/vg150_layouts.jsonl   private layouts (mapped classes,
                                           2-6 mapped objects, area filter)
  data/vg150_private/images/<id>.jpg       extracted images (for rendering)
  data/vg150_private/vocab_map.json        VG150 class -> COCO-80 mapping
Val split (5000 images) = private records; train shards = public pool.

Mapping coverage documented in scripts/check_vg150_mapping.py
(75% of instances, 97% of images keep >=2 mapped objects).
"""
import io
import json
import os
import random
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIN_REL_AREA = 0.005
MIN_OBJECTS, MAX_OBJECTS = 2, 6
TARGET_N = 2859  # matched-N with the COCO private set

SYN = json.load(open(os.path.join(ROOT, "configs", "vg150_to_coco80.json"), encoding="utf-8"))


def main():
    from huggingface_hub import hf_hub_download

    cats_p = hf_hub_download("maelic/VG150-coco-format", "categories.json", repo_type="dataset")
    cats = json.load(open(cats_p, encoding="utf-8"))
    vg_cats = {c["id"]: c["name"] for c in cats["categories"]}
    # prefer a locally uploaded parquet (data/vg150_val.parquet) if COMPLETE;
    # fall back to HF fetch otherwise. Guards against catching a partial
    # SFTP upload mid-flight.
    local_p = os.path.join(ROOT, "data", "vg150_val.parquet")
    val = None
    if os.path.exists(local_p) and os.path.getsize(local_p) > 700e6:
        try:
            val = pd.read_parquet(local_p)
            print("using locally uploaded parquet:", local_p)
        except Exception as e:
            print(f"local parquet unreadable ({e}) -> falling back to HF", flush=True)
            val = None
    if val is None:
        val_p = hf_hub_download("maelic/VG150-coco-format", "data/val-00000-of-00001.parquet",
                                repo_type="dataset")
        val = pd.read_parquet(val_p)

    out_dir = os.path.join(ROOT, "data", "vg150_private")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    dropped = {"few_mapped": 0, "object_count": 0}
    for _, row in val.iterrows():
        W, H = float(row["width"]), float(row["height"])
        img_id = int(row["image_id"])
        objs = []
        for o in row["objects"]:
            target = SYN.get(str(vg_cats[o["category_id"]]).lower())
            if not target:
                continue
            x, y, w, h = (float(v) for v in o["bbox"])
            w, h = min(w, W), min(h, H)
            if (w * h) / float(W * H) < MIN_REL_AREA:
                continue
            objs.append({"class": target, "box": [x / W, y / H, w / W, h / H]})
        if not (MIN_OBJECTS <= len(objs) <= MAX_OBJECTS):
            dropped["object_count"] += 1
            continue
        # dedupe classes within an image? no — keep multiset (matches COCO prep)
        records.append({"image_id": img_id, "caption": "", "objects": objs})
        # extract the image bytes
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            with open(img_path, "wb") as f:
                f.write(row["image"]["bytes"])

    random.Random(0).shuffle(records)
    records = records[:TARGET_N]

    jsonl = os.path.join(out_dir, "vg150_layouts.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    used_classes = sorted({o["class"] for r in records for o in r["objects"]})
    with open(os.path.join(out_dir, "vocab_map.json"), "w", encoding="utf-8") as f:
        json.dump({"used_classes": used_classes, "mapping": SYN}, f, indent=2)

    from collections import Counter
    counts = Counter(len(r["objects"]) for r in records)
    print(f"records: {len(records)} (target {TARGET_N})")
    print(f"objects/image: {dict(sorted(counts.items()))}")
    print(f"n_classes used: {len(used_classes)}")
    print(f"dropped: {dropped}")
    print(f"wrote {jsonl}")


if __name__ == "__main__":
    main()
