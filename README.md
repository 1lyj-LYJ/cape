# CA-PE

Code for "Differentially Private Object Layout Synthesis via Correspondence-Aware
Candidate Assignment".

Training-free differentially private synthesis of object layouts (class +
bounding-box sets) on the Private Evolution (PE) framework. Candidate assignment
matches objects one-to-one with a padded cardinality constraint, so each private
layout casts exactly one vote; per-round voting sensitivity is 1, and the Gaussian
mechanism with Renyi accounting gives an image-level (epsilon, delta) guarantee.

## Layout

    layout/       layout distances and operators: one-to-one matching with padded
                  cardinality costs, Chamfer and optimal-transport baselines,
                  mutation operators, box representation, spatial relations
    pe/           PE engine: voting, Gaussian mechanism + Renyi-DP accountant,
                  stratified parent selection, public candidate pool
    evaluation/   the six structure metrics (Count, Class, Relation, Center,
                  Size, Scale) and joint-structure evaluation
    data/         COCO and VG150 preparation scripts
    configs/      COCO-80 vocabulary, VG150 -> COCO-80 mapping
    scripts/      experiment drivers (see the map below)
    results/      JSON outputs behind the paper's tables and figures
    tests/        unit tests for matching, baselines, privacy boundary

## Requirements

Python 3.10+, then:

    pip install -r requirements.txt

## Data

No data is included; everything is built from public datasets.

1. COCO: download train2017 images and annotations
   (instances_train2017.json) plus val2017 annotations from
   https://cocodataset.org, put them under data/raw/annotations/, and run
   python data/prepare_coco.py. This produces the simulated private set
   data/coco_private/coco_layouts.jsonl (2,859 layouts, images with 2-6
   objects).
2. VG150: python data/prepare_vg150.py maps Visual Genome scene graphs onto
   the COCO-80 vocabulary and drops images that overlap the public candidate
   source (2,409 layouts).
3. The public candidate pool is built from COCO train2017 statistics only and
   never touches the private val2017 records.

## Reproduction

    python scripts/unified_main_table.py      # Tables 1-2, T=1, eps=5, 10 seeds
    python scripts/ablation_noisy.py          # Fig. 3 component ablation
    python scripts/eps_curve_5pt.py           # Fig. 4, CA-PE curve
    python scripts/eps_curve_baselines.py     # Fig. 4, +Card baselines
    python scripts/lambda_class_sweep.py      # lambda_class trade-off, Sec. 4.3.3
    python scripts/pool_scarcity_seeds.py     # K=200 multi-round, Sec. 4.4.2
    python scripts/classcond_relation.py      # class-conditional relations, 4.2.2
    python scripts/lc_release.py              # downstream releases, Sec. 4.5
    python scripts/lc_task2.py                # layout completion, Sec. 4.5
    python scripts/lc_vg.py                   # VG150 downstream, Table 2

Scripts resolve paths from the repository root, write outputs under runs/, and
use fixed randomness: 10 outer noise seeds (0-9), 12 release resamples per seed,
inner variation seeds {0,1,2}.

results/ holds the JSON files cited by the paper, from the code state frozen on
2026-10-28.

## License

MIT.
