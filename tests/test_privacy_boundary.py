"""Static privacy-boundary check: post-processing modules must not touch
private data. Run: py tests/test_privacy_boundary.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# modules that are (part of) the DP release path — they may consume noisy
# votes and PUBLIC candidates only
RELEASE_PATH = [
    "pe/selection.py",
    "layout/mutation.py",
    "layout/projection.py",
    "layout/representation.py",
    "layout/matching.py",
    "layout/relations.py",
    "layout/distance.py",
    "generation/gligen.py",
]

# tokens that would indicate private-data access inside the release path
FORBIDDEN = [
    "coco_layouts.jsonl",      # private layout file
    "prepare_coco",            # private prep entry point
    "instances_val2017",       # private annotation file
    "captions_val2017",
    "load_private",            # private loader helper
    "vote_matrix",             # raw-vote computation
    "true_votes",              # raw votes
]

failures = []
for rel in RELEASE_PATH:
    path = os.path.join(ROOT, rel)
    src = open(path, encoding="utf-8").read()
    for tok in FORBIDDEN:
        if tok in src:
            failures.append(f"{rel}: contains forbidden token '{tok}'")

if failures:
    print("PRIVACY BOUNDARY VIOLATIONS:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print(f"privacy boundary OK: {len(RELEASE_PATH)} release-path modules clean "
      f"({len(FORBIDDEN)} forbidden tokens checked)")
