# Argus Localization

Retrieve-then-match localizer for Argus: retrieval (EarthLoc-style) picks a
shortlist of candidate reference tiles, then a feature matcher (SIFT-LightGlue,
EarthMatch-style) verifies geometric overlap and produces pixel-to-ground tie
points. Open-set by construction: a weak match is rejected instead of silently
feeding bad measurements downstream.

Full rationale: [docs/argus_localization_design.md](docs/argus_localization_design.md)
Implementation spec (source of truth for interfaces): [docs/argus_localization_spec.md](docs/argus_localization_spec.md)

Status: Phase 0 done. Retriever, index, matcher, georeferencer, reference
database, and the pipeline are implemented and run end to end on the rsynced
EarthLoc data. Phase 1 (Landsat-8 reference DB at Argus scale, source
imagery at `/mnt/sda2/geotiffs/`, 16 MGRS zones; supersedes the "Sentinel-2"
wording in the design/spec docs) is next, see spec section 9.

Corner order note: `GeoTile.corners_latlon` is `BL,TL,TR,BR`, verified
against the real EarthLoc filenames. This differs from the `TL,TR,BR,BL`
order in the original spec draft, which was wrong. `demo.py` renders a
corner-labeled sanity check image so this stays checkable.

## Layout

```
core/            GeoTile, TiePoint, LocalizationResult, MatchResult, PipelineConfig
                 Retriever / DescriptorIndex / Matcher protocols, LocalizationPipeline
data_loading/    EarthLoc filename parsing, query/reference tile loaders
retrievers/      EarthLocRetriever (v0 baseline), SmallRetriever (phase 2 target)
index/           FaissFlatIndex
matchers/        SiftLightGlueMatcher
georeference/    pixel -> lat/lon over a tile's four corners
database/        ReferenceDatabase (embed, index, retrieve)
integration/     to_batchopt_measurements, the one function that touches OD
training/        train_retriever.py, phase 2 distillation stub, not run in v0
scripts/         evaluate.py (recall@k, fix rate, localization error), demo.py
docs/            design doc and implementation spec
```

## Data

Reference tiles and queries are rsynced from [EarthLoc](https://github.com/gmberton/EarthLoc)
into `/mnt/sdc1/astroloc/data` (read-only on this machine, owned by root):

- `queries/*.jpg`, 17763 files
- `database/YYYY_MM/*.jpg`
- `best_trained_model.pt`: released EarthLoc checkpoint (DINOv2-base + SALAD)

Paths are in `user_config.yaml`, not hardcoded, since they differ per machine.

## Setup

```
pip install -r requirements.txt
pip install shapely
pip install git+https://github.com/cvg/LightGlue.git
pip install -e .
```

`third_party/EarthLoc/` is a vendored, read-only, gitignored copy of
https://github.com/gmberton/EarthLoc, used two ways: `scripts/reproduce_earthloc_recall.py`
imports it freely to reproduce EarthLoc's own numbers, and
`retrievers/earthloc_retriever.py` imports only `apl_models.apl_model.APLModel`
from it (nothing else from EarthLoc is allowed into the Argus pipeline). If
`third_party/EarthLoc` does not exist, clone it there:
`git clone https://github.com/gmberton/EarthLoc.git third_party/EarthLoc`.

Paths (data root, checkpoint, cache dir) come from `user_config.yaml`.

## Running Phase 0

```
# Deliverable 1: plumbing check, EarthLoc's own model/dataset/test code, one region.
python scripts/reproduce_earthloc_recall.py --region-name Alps

# Deliverable 2/3: Argus pipeline, builds and caches a reference DB on first run.
python scripts/evaluate.py --region Alps

# Deliverable 3: one annotated demo (draws inlier correspondences and a
# corner-order sanity check), requires evaluate.py to have run once first
# so a cached reference DB exists.
python scripts/demo.py path/to/query.jpg --output output/demo.png
```

EarthLoc queries are oblique and cover about 25,000 sq km each, nothing like
an Argus frame, so these numbers validate plumbing (the model loads, the
data loads, retrieval and matching run end to end), not Argus performance.

`evaluate.py`'s retrieval recall (R@1 ~25, R@15 ~45) is expected to be much
lower than `reproduce_earthloc_recall.py`'s (R@1 58.4, R@100 89.4). Measured
on the Alps region, two things drive essentially all of the gap, and neither
is a retrieval quality bug:

1. Ground truth strictness (about 20% of the gap at R@15). `evaluate.py`
   labels a retrieved tile correct only if its footprint overlaps the query's
   at IoU >= 0.2 (`eval.iou_threshold` in config.yaml, per spec section 7).
   EarthLoc's own precomputed `queries_intersections_with_db_2021.pt` is far
   more generous, about 65 positive tiles per query on average versus about
   10 for our IoU check on the same queries. This is mechanical: EarthLoc
   query footprints (~25,000 sq km) are about a quarter the area of a
   database tile (~97,000-98,000 sq km), so even a query fully contained in
   one tile caps IoU well below 1.0. This is the scale mismatch the design
   doc already flags (section 8, risk 4) as the most likely reason numbers
   look bad for a reason unrelated to the method, and it is the reason
   Phase 1 rebuilds the reference DB at Argus's actual matched scale.
2. Rotation test-time augmentation (the rest of the gap). EarthLoc's own
   `test.py` embeds every database tile at 4 rotations (0/90/180/270) and
   keeps the best-aligned one per query, which matters a lot for oblique,
   arbitrarily-rotated astronaut photography. `EarthLocRetriever`
   deliberately does not do this (Phase 0 build prompt, Deliverable 2), so
   this is an expected difference, not a bug.

Swapping in EarthLoc's own ground truth while still using our non-augmented
retriever recovers only part of the gap (R@15 45 -> 54 on a 400-query
sample), confirming augmentation is the larger factor of the two.

## Performance notes

`EarthLocRetriever.embed_batch` resizes and normalizes the whole batch on
GPU instead of PIL/torchvision per image on CPU (measured ~132 img/s ->
~254 img/s on this machine; the per-image CPU path left the GPU under 20%
utilized per `nvidia-smi`, confirmed with `nvidia-smi dmon`). `evaluate.py`'s
retrieval eval batches query embedding the same way instead of one image at
a time. Further speedup is available by decoding JPEGs on GPU via
`torchvision.io.decode_jpeg(..., device="cuda")` (nvJPEG measured at ~1275
img/s batched), but that would require the retriever to accept file paths
instead of decoded arrays, which breaks the `Retriever.embed_batch(images:
list[np.ndarray])` contract other retrievers (e.g. the Phase 2 small
retriever) also need to satisfy for a live in-memory camera frame, so it was
not done here.

## Next steps

Phase 1: build a reference DB at Argus scale from the Landsat-8 GeoTIFFs at
`/mnt/sda2/geotiffs/` (16 MGRS zones: 10S, 10T, 11R, 12R, 16T, 17R, 17T, 18S,
32S, 32T, 33S, 33T, 52S, 53S, 54S, 54T; each zone has ~500 temporal
composites of the same near-full-zone footprint at 175m/pixel, UTM
projected). This supersedes the "Sentinel-2" wording in the design and spec
docs, confirmed with Priyanka on 2026-07-01. Tiling will need to reproject
each zone's UTM footprint to lat/lon corners for `GeoTile.corners_latlon`
and pick a subset of the ~500 timestamps per zone rather than using all of
them. Confirm retrieval is sane once query and DB scales match (the current
numbers use EarthLoc's own mismatched scales, bootstrap only). Then phase 2
(small distilled retriever), phase 3 (matcher integration into batch opt).
See `docs/argus_localization_spec.md` section 9.
