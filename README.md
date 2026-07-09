# Argus Localization

A prototype "retrieve-then-match" image localizer, built to replace the closed-set RCNet/YOLO region and landmark detection stage in Argus's spacecraft orbit-determination (OD) pipeline.

Given a camera frame, it answers two questions in sequence:

1. **Roughly where is this?** A learned image retriever embeds the frame and searches a reference database of geo-tagged satellite tiles for the most visually similar ones (EarthLoc-style).
2. **Exactly where, and do I trust it?** A classical feature matcher (SIFT + LightGlue, EarthMatch-style) checks each retrieved candidate for real geometric overlap with the frame, via keypoint correspondences and a fitted homography. The inlier count from that fit is the confidence score.

If nothing matches well enough, the pipeline reports `no_fix` instead of guessing. That is the whole point of this design: RCNet/YOLO are closed-set and silently fail on frames unlike their training data (oblique views, unfamiliar terrain), which corrupts OD downstream with no "I don't know" option. Retrieval similarity plus matcher inlier count give a number that can be thresholded, so weak matches get rejected instead of silently fed forward as bad measurements.

Full rationale: [docs/argus_localization_design.md](docs/argus_localization_design.md)
Implementation spec, source of truth for interfaces: [docs/argus_localization_spec.md](docs/argus_localization_spec.md)

## How it works

```
                 ┌────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌────────────────┐
   camera  ───▶  │  Retriever │──▶│ DescriptorIndex│──▶│     Matcher      │──▶│  Georeferencer │──▶ LocalizationResult
   frame         │ (EarthLoc) │   │  (faiss, top-k)│   │ (SIFT+LightGlue) │   │ (pixel->latlon)│      (tie points,
                 └────────────┘   └───────────────┘   └──────────────────┘   └────────────────┘       footprint,
                                                                                                        confidence)
```

Every stage is a `typing.Protocol` (`core/interfaces.py`), so any concrete implementation can be swapped without touching the pipeline. This is what lets the project start with the released EarthLoc retriever for plumbing, then later drop in a small distilled retriever, without rewriting anything downstream.

### Stage by stage

**1. Retriever** (`core/interfaces.py::Retriever`, implemented in `retrievers/earthloc_retriever.py::EarthLocRetriever`)
Wraps the released EarthLoc checkpoint (ResNet50 backbone + MixVPR aggregation, see `third_party/EarthLoc/apl_models/`). Takes an `(H,W,3)` uint8 image, returns a 4096-dim L2-normalized descriptor vector. This is a baseline for plumbing only, not the deployment retriever: it is a ~400MB model trained on astronaut photography, not something that runs on flight hardware. The Phase 2 target, `retrievers/small_retriever.py::SmallRetriever`, is a stub for a distilled EfficientNet-Lite/MobileNetV3 model with a 512-dim descriptor, not implemented yet.

**2. Index** (`core/interfaces.py::DescriptorIndex`, implemented in `index/faiss_index.py::FaissFlatIndex`)
A flat FAISS `IndexFlatIP` over L2-normalized descriptors, which gives exact cosine similarity search. `database/reference_database.py::ReferenceDatabase` is the layer above this: it embeds every reference tile once (`build()`), stores tile metadata (`GeoTile` objects keyed by `tile_id`), and answers `retrieve(frame, k)` by embedding the query and searching the index. This embed-once, search-many-times database is "the dataset" that retrieval runs against; it is cached to disk under `cache/db_<region>/` so it does not need rebuilding on every run.

**3. Matcher** (`core/interfaces.py::Matcher`, implemented in `matchers/sift_lightglue_matcher.py::SiftLightGlueMatcher`)
For a (query frame, candidate tile) pair: extracts SIFT keypoints from both (max 1024 keypoints, images resized to 512px), matches them with LightGlue, then fits a homography with `cv2.findHomography` using MAGSAC RANSAC. Returns a `MatchResult`: the matched pixel coordinates in both images, an inlier mask, the homography, and `num_inliers`. This mirrors the EarthMatch algorithm (see "EarthMatch reproduction" below). The matcher does not decide accept/reject; it just reports the inlier count. Database-side keypoints are cached per `tile_id` so repeated candidates across queries do not re-extract features.

**4. Pipeline orchestration** (`core/pipeline.py::LocalizationPipeline.localize()`)
Retrieves the top-`k` candidates (default 15), runs the matcher against every one of them, and keeps whichever candidate got the most inliers. If the winner's inlier count is below `min_inliers` (default 30), returns `status="no_fix"` with no tie points. Otherwise, returns `status="fix"` with the confidence (inlier count), the matched tile id, and the georeferenced output below.

**5. Georeferencer** (`georeference/georeferencer.py::Georeferencer`)
Pure numpy, no model. Reference tiles are nadir (looking straight down), so their four corner lat/lon coordinates (`GeoTile.corners_latlon`) define a regular grid; any pixel inside the tile gets its lat/lon by bilinear interpolation over that grid. Two things get computed this way: `tie_points` (every inlier match, as `(query pixel u,v) <-> (lat, lon)`), and `query_footprint_latlon` (the query frame's own four corners, warped through the fitted homography into tile-pixel space, then interpolated to lat/lon). Corner order is `BL, TL, TR, BR`, verified empirically against real EarthLoc filenames (this differs from an earlier, wrong assumption in the original spec draft; `scripts/demo.py` renders a corner-labeled sanity image so this stays checkable).

### Core data types (`core/types.py`)

- `GeoTile`: a reference tile or a query image. `tile_id`, `image_path`, `corners_latlon` (4x2 array), optional `timestamp`/`meta`.
- `MatchResult`: what the matcher returns. Matched pixel coordinates in both images, inlier mask, homography, inlier count.
- `TiePoint`: one georeferenced correspondence, `(u, v, lat, lon)`.
- `LocalizationResult`: what the pipeline returns. `status` (`"fix"` or `"no_fix"`), `confidence`, `tie_points`, `matched_tile_id`, `query_footprint_latlon`, and a `debug` dict with per-candidate scores.
- `PipelineConfig`: `top_k`, `min_inliers`, `query_size`, `max_ransac_iters`.

### Where OD fits in (deliberately not built yet)

`integration/batchopt_adapter.py::to_batchopt_measurements` is the **only** function meant to touch OD-specific formatting (field names, coordinate frame, covariance weighting). It is currently a stub that raises `NotImplementedError` on purpose: OD integration is a later phase (see spec section 8, and the phased plan below). Everything upstream of that one function is finished and stays fixed regardless of when or how OD integration happens; only that adapter changes.

## Repo layout

```
core/            GeoTile, TiePoint, LocalizationResult, MatchResult, PipelineConfig
                 Retriever / DescriptorIndex / Matcher protocols, LocalizationPipeline
data_loading/    EarthLoc filename parsing, query/reference tile loaders
retrievers/      EarthLocRetriever (v0 baseline, implemented), SmallRetriever (phase 2 target, stub)
index/           FaissFlatIndex
matchers/        SiftLightGlueMatcher
georeference/    pixel -> lat/lon over a tile's four corners
database/        ReferenceDatabase (embed, index, retrieve, save/load cache)
integration/     to_batchopt_measurements, the one function that touches OD (stub, Phase 3)
training/        train_retriever.py, phase 2 distillation stub, not run in v0
scripts/         reproduce_earthloc_recall.py, evaluate.py, demo.py, build_coordinates_dataset.py
docs/            design doc and implementation spec
third_party/     vendored, gitignored, read-only copy of github.com/gmberton/EarthLoc
cache/           gitignored. Cached FAISS indexes and tile metadata per region, db_<region>/
output/          gitignored. Demo images and coordinates_<region>.json datasets
```

## Current status

Everything in the diagram above is implemented and runs end to end, not stubbed:
`EarthLocRetriever`, `FaissFlatIndex`, `SiftLightGlueMatcher`, `Georeferencer`, `ReferenceDatabase`, `LocalizationPipeline`.

Still a stub, on purpose, gated on later phases:
`SmallRetriever` and `training/train_retriever.py` (Phase 2), `integration/batchopt_adapter.py` (Phase 3+, OD integration).

Everything has only ever been run against the rsynced EarthLoc dataset (astronaut photography), not real Argus imagery. That data is bootstrap only: EarthLoc queries are oblique and cover about 25,000 sq km each, nothing like a near-nadir Argus frame. The numbers below validate that the plumbing works end to end, not what accuracy Argus should expect in production. Phase 1 (next up) rebuilds the reference database at Argus's actual scale to get a real answer to that.

## Data

Reference tiles and queries are rsynced from [EarthLoc](https://github.com/gmberton/EarthLoc) into `/mnt/sdc1/astroloc/data` (read-only on this machine, owned by root):

- `queries/*.jpg`, 17763 files, each an oblique astronaut photograph
- `database/YYYY_MM/*.jpg`, nadir satellite tiles, multiple years
- `best_trained_model.pt`, the released EarthLoc checkpoint (DINOv2-base + SALAD)

Both queries and reference tiles encode their footprint and metadata directly in the filename:

```
@lat1@lon1@lat2@lon2@lat3@lon3@lat4@lon4@image_id@timestamp@nadir_lat@nadir_lon@sq_km_area@orientation@.jpg
```

`data_loading/earthloc_loader.py::parse_geotile_filename` parses this into a `GeoTile`. Paths themselves live in `user_config.yaml`, not hardcoded, since they differ per machine.

## Setup

```
pip install -r requirements.txt
pip install shapely
pip install git+https://github.com/cvg/LightGlue.git
pip install -e .
```

`third_party/EarthLoc/` is a vendored, read-only, gitignored copy of https://github.com/gmberton/EarthLoc, used two ways: `scripts/reproduce_earthloc_recall.py` imports it freely to reproduce EarthLoc's own numbers, and `retrievers/earthloc_retriever.py` imports only `apl_models.apl_model.APLModel` from it (nothing else from EarthLoc is allowed into the Argus pipeline itself). If `third_party/EarthLoc` does not exist, clone it there:
`git clone https://github.com/gmberton/EarthLoc.git third_party/EarthLoc`.

## Running it

```
# Plumbing check: reproduce EarthLoc's own reported recall, using EarthLoc's own
# model, dataset, and test code directly (not the Argus pipeline). Proves the
# released checkpoint and the rsynced data load and run correctly.
python scripts/reproduce_earthloc_recall.py --region-name Alps

# Argus pipeline end to end: builds and caches a reference database on first
# run (slow, has to embed every reference tile), then reports retrieval
# recall@k and matching fix rate / localization error.
python scripts/evaluate.py --region Alps

# One annotated demo: draws inlier correspondences between a query and its
# matched tile, plus a corner-order sanity check image. Requires evaluate.py
# to have run once first so a cached reference database exists.
python scripts/demo.py path/to/query.jpg --output output/demo.png

# Batch export: run the full pipeline over many frames and write out a
# coordinates dataset (tie points + estimated footprint per frame), the
# input to OD once integration/batchopt_adapter.py is implemented. Defaults
# to the region-scoped EarthLoc query set; pass --frames-dir for any folder
# of images instead.
python scripts/build_coordinates_dataset.py --region Alps --limit 50
```

All four scripts read `config.yaml` (pipeline/matcher/eval hyperparameters, checked into git) and `user_config.yaml` (machine-local paths, not meant to be identical across dev machines).

## Current results

### EarthLoc reproduction (`reproduce_earthloc_recall.py`, Alps region)

This is EarthLoc's own retrieval model and test harness, run inside this repo purely as a plumbing check (does the checkpoint load, does the data load):

```
R@1: 58.4, R@5: 72.3, R@10: 76.9, R@20: 81.0, R@100: 89.4
```

This matches EarthLoc's own published numbers for this region, confirming the checkpoint and data are wired correctly.

### Argus pipeline evaluation (`evaluate.py`, Alps region)

```
=== Retrieval (Alps, 2393 queries with a ground-truth positive) ===
R@1: 25.4, R@5: 35.9, R@15: 45.0

=== Matching (Alps, 50 queries) ===
fix rate (num_inliers >= 30): 36.0%
median localization error: 3.3 km
```

Retrieval recall here is much lower than the EarthLoc reproduction above (R@1 25.4 vs 58.4). This gap has been measured and understood, it is not a retrieval quality bug:

1. **Ground truth strictness (about 20% of the gap at R@15).** `evaluate.py` only counts a retrieved tile as correct if its footprint overlaps the query's at IoU >= 0.2. EarthLoc's own precomputed positives file is far more generous (about 65 positive tiles per query on average, versus about 10 under our IoU check on the same queries), because EarthLoc query footprints (~25,000 sq km) are only about a quarter the area of a database tile (~97,000-98,000 sq km): even a query fully contained in one tile caps IoU well below 1.0. This is the query/database scale mismatch the design doc already flags as the most likely reason numbers look bad for a reason unrelated to the method, and it is exactly why Phase 1 rebuilds the reference database at Argus's actual matched scale.
2. **Rotation test-time augmentation (the rest of the gap).** EarthLoc's own test code embeds every database tile at four rotations (0/90/180/270) and keeps whichever aligns best with the query, which matters a lot for oblique, arbitrarily-rotated astronaut photography. `EarthLocRetriever` deliberately does not do this. Swapping in EarthLoc's own generous ground truth while still using our non-augmented retriever only recovers part of the gap (R@15 45 -> 54 on a 400-query sample), confirming augmentation is the larger of the two factors.

### Coordinates dataset export (`build_coordinates_dataset.py`, all 6 EarthLoc regions, 50 frames each)

```
Region          Reference tiles embedded   Fix rate
Alps            52951                      36.0%
Texas           34032                      20.0%
Toshka Lakes    62617                      46.0%
Amazon          19126                      28.0%
Napa            30400                      28.0%
Gobi            54687                      32.0%
```

Fix rate is the fraction of sampled frames where the best-matching candidate cleared the 30-inlier threshold and produced real coordinates. The spread across regions (20-46%) tracks terrain and texture, not a bug: Toshka Lakes' high-contrast desert/water coastlines give SIFT strong, distinctive keypoints, while Texas' more uniform agricultural/urban texture at this scale gives fewer distinctive features to match confidently. As above, these numbers describe matching EarthLoc's oblique, large-area astronaut photography against a mismatched-scale reference database, not Argus's own frames; they confirm the retrieve-match-georeference pipeline runs correctly end to end and produces sane tie points and footprints (spot-checked against ground truth), not what fix rate to expect in production.

### EarthMatch reproduction (separate repo, sanity check only)

The matcher here (`SiftLightGlueMatcher`) mirrors the algorithm from [EarthMatch](https://github.com/gmberton/EarthMatch) (SIFT + LightGlue + iterative RANSAC coregistration). Running EarthMatch's own released code and benchmark data (268 astronaut photo queries with precomputed top-10 retrieval candidates) reproduced its qualitative behavior (zero false positives surviving the iterative filter, i.e. `threshold=-1`) but landed about 8 points below its published headline number (81.1% located vs. 89.3% reported), with per-subset swings in both directions. The likely cause: `cv2.findHomography`'s RANSAC has its own internal, unseeded random state, so which borderline queries survive all iterations can vary run to run; this was not root-caused further since EarthMatch is an external reference, not code in this repo. Worth keeping in mind as a source of run-to-run variance in this repo's own matcher too, since it uses the same RANSAC call.

## Performance notes

`EarthLocRetriever.embed_batch` resizes and normalizes the whole batch on GPU instead of doing PIL/torchvision per image on CPU (measured ~132 img/s -> ~254 img/s on this machine; the per-image CPU path left the GPU under 20% utilized per `nvidia-smi dmon`). `evaluate.py`'s retrieval eval batches query embedding the same way instead of one image at a time.

The matcher (SIFT + LightGlue + RANSAC) is the slow stage, not retrieval: roughly 1-1.5 sec/frame, dominated by CPU-side classical SIFT keypoint extraction rather than GPU-side LightGlue matching. This is why `evaluate.py` and `build_coordinates_dataset.py` both subsample or cap the number of frames sent through matching (`eval.max_matching_queries` in `config.yaml`, or `--limit`), while retrieval alone can run over every query cheaply.

Further retrieval speedup is available by decoding JPEGs on GPU via `torchvision.io.decode_jpeg(..., device="cuda")` (nvJPEG measured at ~1275 img/s batched), but that would require the retriever to accept file paths instead of decoded arrays, breaking the `Retriever.embed_batch(images: list[np.ndarray])` contract that other retrievers (including the Phase 2 small retriever) also need to satisfy for a live in-memory camera frame, so it was not done here.

## Phased plan

Gate each phase on the previous one succeeding.

- **Phase 0 (done):** wire the released EarthLoc retriever, FaissFlatIndex, and a SIFT-LightGlue matcher into one pipeline; get baseline recall and fix-rate numbers on the rsynced EarthLoc data. Validates plumbing only, not Argus-representative accuracy (see caveats above).
- **Phase 1 (next):** rebuild the reference database at Argus's actual scale from the Landsat-8 GeoTIFFs at `/mnt/sda2/geotiffs/` (16 MGRS zones: 10S, 10T, 11R, 12R, 16T, 17R, 17T, 18S, 32S, 32T, 33S, 33T, 52S, 53S, 54S, 54T; each zone has about 500 temporal composites of the same near-full-zone footprint at 175m/pixel, UTM projected). This supersedes the "Sentinel-2" wording still present in the design and spec docs. Tiling needs to reproject each zone's UTM footprint to lat/lon corners for `GeoTile.corners_latlon`, and pick a subset of the roughly 500 timestamps per zone rather than using all of them. This is a make-or-break check: confirm retrieval is sane once query and database scales actually match, since the numbers above use EarthLoc's own mismatched scales.
- **Phase 2:** distill a small on-device retriever (EfficientNet-Lite/MobileNetV3 + GeM, descriptor_dim 512) from the EarthLoc DINOv2+SALAD teacher via relational (pairwise-similarity) distillation, then domain fine-tune on the Phase 1 reference tiles. Goal is recall@k parity with the teacher, not matching its R@1 exactly.
- **Phase 3:** implement `integration/batchopt_adapter.py::to_batchopt_measurements` for real, and integrate the matcher's tie points into the batch optimization step of OD. End-to-end localization quality becomes the metric that matters.
- **Phase 4:** Orin hardware benchmarks (median and worst-case frame time vs. the current RCNet/YOLO path), scene-stratified quality. Textureless scenes (ocean, uniform desert, ice, thick cloud) are explicitly out of scope per current project scoping, so there is no YOLO fallback in v0; this is a team decision to revisit if those scenes turn out to be in the operating population.

See `docs/argus_localization_spec.md` section 9 for the full detail behind each phase.
