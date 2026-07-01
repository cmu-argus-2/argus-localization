# Argus Localization: Implementation Spec (v0 prototype)

Companion to `argus_localization_design.md` (the rationale doc). That doc is the "why" for the team. This is the "how" for a coding agent. Where they disagree, the rationale doc wins on intent; this doc wins on interfaces.

Hard rule for all code and comments: no em dashes anywhere.

Update (2026-07-01, confirmed with Priyanka): every "Sentinel-2" mention below for the Phase 1 reference DB is superseded. The actual available imagery on this machine is Landsat-8 (`/mnt/sda2/geotiffs/`, 16 MGRS zones, `l8_` filename prefix, B4/B3/B2 bands, 175m/pixel, ~500 temporal composites per zone). Read "Sentinel-2" as "Landsat-8" throughout until this doc is revised.

## 0. Goal of v0

Produce a runnable, demonstrable retrieve-then-match localizer that runs on the existing EarthLoc-format data, reports metrics, and renders one demo. Do NOT integrate into flight/OD code. Do NOT train anything in v0. Get plumbing and numbers first (this is Phase 0/1 of the rationale doc).

## 1. Pipeline

```
frame --> Retriever --> descriptor --> Index.search(k) --> top-k tiles
      --> for each tile: Matcher.match(frame, tile) --> correspondences + inliers + homography
      --> Georeferencer --> (u,v) -> (lat,lon) tie points
      --> pick max-inlier candidate, threshold on inliers --> LocalizationResult
```

Retrieval answers "roughly where." Matching answers "exactly which pixels map to which ground points, and do I trust it." Confidence is the inlier count.

## 2. Modularity

Every stage is a `typing.Protocol` (or ABC). Retriever, Index, Matcher, ReferenceDatabase, Georeferencer are independently swappable. The pipeline is stateless per call: frame in, `LocalizationResult` out. This is what lets us start with the released EarthLoc retriever and later drop in a small distilled one without touching anything else.

## 3. Core types

```python
from dataclasses import dataclass
from typing import Protocol, Optional
import numpy as np

@dataclass(frozen=True)
class GeoTile:
    tile_id: str
    image_path: str
    corners_latlon: np.ndarray        # (4,2) lat,lon; order matches pixel corners TL,TR,BR,BL
    timestamp: Optional[str] = None
    meta: Optional[dict] = None

@dataclass(frozen=True)
class TiePoint:
    u: float; v: float                # query pixel
    lat: float; lon: float            # world

@dataclass(frozen=True)
class LocalizationResult:
    status: str                       # "fix" | "no_fix"
    confidence: float                 # inlier count (or normalized)
    tie_points: list[TiePoint]
    matched_tile_id: Optional[str]
    query_footprint_latlon: Optional[np.ndarray]   # (4,2) or None
    debug: Optional[dict] = None      # per-candidate scores, timings
```

## 4. Interfaces

```python
class Retriever(Protocol):
    descriptor_dim: int
    def embed(self, image: np.ndarray) -> np.ndarray: ...          # (H,W,3)uint8 -> (D,)f32 L2-normed
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray: # -> (N,D)f32 L2-normed

class DescriptorIndex(Protocol):
    def add(self, tile_ids: list[str], descriptors: np.ndarray) -> None: ...
    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]: ...   # (tile_id, sim), best first
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

@dataclass(frozen=True)
class MatchResult:
    query_pts: np.ndarray             # (M,2)
    tile_pts: np.ndarray              # (M,2)
    inlier_mask: np.ndarray           # (M,) bool
    homography: Optional[np.ndarray]  # (3,3) query->tile or None
    num_inliers: int

class Matcher(Protocol):
    def match(self, query_image: np.ndarray, tile_image: np.ndarray,
              tile_id: Optional[str] = None) -> MatchResult: ...
    # tile_id lets an implementation use precomputed database-side keypoints.

class Georeferencer:
    def tile_pixel_to_latlon(self, tile: GeoTile, tile_shape, pixels: np.ndarray) -> np.ndarray: ...
    def make_tie_points(self, match: MatchResult, tile: GeoTile, tile_shape) -> list[TiePoint]: ...
    def estimate_query_footprint(self, match, tile, query_shape, tile_shape) -> Optional[np.ndarray]: ...

@dataclass
class PipelineConfig:
    top_k: int = 15                   # start high, measure recall@k, trade down
    min_inliers: int = 30
    query_size: int = 512
    max_ransac_iters: int = 3

class LocalizationPipeline:
    def __init__(self, db, matcher, georef, config): ...
    def localize(self, frame: np.ndarray) -> LocalizationResult: ...

class ReferenceDatabase:
    def __init__(self, retriever: Retriever, index: DescriptorIndex): ...
    def build(self, tiles: list[GeoTile]) -> None: ...   # embed all, populate index, cache to disk
    def get_tile(self, tile_id: str) -> GeoTile: ...
    def retrieve(self, frame: np.ndarray, k: int) -> list[tuple[GeoTile, float]]: ...
    def save(self, dir_path: str) -> None: ...
    @classmethod
    def load(cls, dir_path, retriever, index) -> "ReferenceDatabase": ...
```

## 5. v0 concrete implementations (all behind the interfaces above)

- Retriever, v0 default: wrap the released EarthLoc model (`best_trained_model.pt`, DINOv2-base + SALAD) as `EarthLocRetriever(Retriever)`. This is only to validate plumbing and get baseline recall fast. It is NOT the deployment retriever.
- Retriever, Phase 2 target: `SmallRetriever(Retriever)` = timm EfficientNet-Lite or MobileNetV3 + GeM pooling + L2 norm, descriptor_dim 512, distilled from the EarthLoc teacher with relational (pairwise-similarity) distillation, then domain fine-tuned on Sentinel tiles. Leave `train_retriever.py` as a stub with the distillation loss sketched, do not run it in v0.
- Index: `FaissFlatIndex(DescriptorIndex)` over `IndexFlatIP` (cosine on normalized vectors). Small DB, exact search. Note IVF-PQ as the later swap if RAM-bound.
- Matcher: `SiftLightGlueMatcher(Matcher)`. LightGlue with SIFT features (mirror EarthMatch). Fit homography with `cv2.USAC_MAGSAC` or `cv2.RANSAC`, inlier mask from that. Lean config: `max_num_keypoints=1024`, `img_size=512`, hard cap `max_ransac_iters=3`, fail-fast if inliers < min_inliers. Support precomputed database-side keypoints keyed by tile_id.
- Georeferencer: pure numpy, bilinear interpolation over the tile's four corners (database tiles are nadir, treat as a regular grid).

Deps: torch, timm, faiss-cpu, lightglue (or kornia's LightGlue), opencv-python, numpy, pillow.

## 6. Data loading (bootstrap on existing EarthLoc data)

Reference tiles at `database/YYYY_MM/*.jpg`, queries at `queries/*.jpg`. Filename schema for both:

```
@lat1@lon1@lat2@lon2@lat3@lon3@lat4@lon4@image_id@timestamp@nadir_lat@nadir_lon@sq_km_area@orientation@.jpg
```

The four lat/lon pairs are the footprint corners. Write `earthloc_loader.py`:

```python
def parse_geotile_filename(path: str) -> GeoTile: ...
def load_reference_tiles(database_dir: str) -> list[GeoTile]: ...
def load_query_set(queries_dir: str) -> list[GeoTile]: ...   # queries carry footprints = ground truth
```

Keep the loader generic. Argus-simulated near-nadir queries (rendered from Sentinel over MGRS regions) must drop in through the same `GeoTile` interface. The EarthLoc queries are oblique and large-area (about 25,000 sq km each), so they are for smoke-testing only, not Argus-representative evaluation.

## 7. Eval + demo harness

`evaluate.py`:
- Retrieval: report recall@1, recall@5, recall@k. A retrieval counts as correct if a retrieved tile footprint overlaps the query ground-truth footprint above IoU 0.2 (configurable). recall@k is the metric that matters, NOT R@1 (see rationale doc section 6).
- Matching: per query, best-candidate num_inliers, fix rate (frac with num_inliers >= min_inliers), median localization error in km (tie-point centroid or footprint center vs ground truth).
- Report retrieval and matching separately. They fail for different reasons.

`demo.py`: one frame in, render the retrieved tile, draw inlier correspondences between frame and tile, print/plot the estimated footprint. This is the visual for the prof.

## 8. Integration point (the only thing that touches OD)

Downstream consumes `LocalizationResult.tie_points`: (image u, image v) paired with (lat, lon). This is the same image-to-world landmark set the current landmark-detection stage feeds to batch optimization. Isolate the mapping in ONE function:

```python
def to_batchopt_measurements(result: LocalizationResult):
    """TODO(team): map tie_points into batch-opt's measurement format.
    Confirm: field names, coordinate frame (lat/lon vs ECEF), and covariance/weight
    per correspondence. This is the ONLY function that changes at integration time.
    Everything upstream stays fixed."""
    raise NotImplementedError
```

Optional hybrid mode (only if oceans/deserts turn out to be in scope, see non-goals): on `no_fix`, route the frame to the existing YOLO landmark set as a fallback, and also emit a tile-to-region lookup for RCNet routing. Keep this OFF by default in v0.

## 9. Phases (gate each on the previous)

- Phase 0: released EarthLoc retriever + matcher + harness on the rsync data. Baseline recall on the six EarthLoc regions. Validates plumbing, not Argus performance.
- Phase 1: build a Sentinel-2 reference DB at Argus scale via existing MGRS tiling. Confirm retrieval is sane when query and DB scales match. Make-or-break check.
- Phase 2: distill + narrow-train the small retriever, measure recall@k vs the EarthLoc teacher, pick the smallest model that holds recall@k.
- Phase 3: matcher over top-k into batch opt, end-to-end localization quality.
- Phase 4: Orin benchmarks (median + worst-case frame time, old path vs new), scene-stratified quality, INT8 retriever, optional pose-prior seeding of the geometric fit.

## 10. Assumptions and non-goals

- Assumption: query and reference tiles are at comparable ground scale. Poor retrieval is a scale-mismatch suspect before it is a model suspect.
- Assumption (from latest scoping): regions of interest are coastlines and high-saliency land. Ocean, uniform desert, ice, and thick cloud are OUT of the operating population, so v0 does not handle textureless-scene matcher failure and ships matcher-primary with no fallback. NOTE: the rationale doc treats textureless scenes as a live risk and proposes a YOLO fallback hybrid. If the team decides those scenes can occur, flip on the optional hybrid in section 8. This is a team decision.
- Non-goals for v0: OD integration, training, quantization, flight constraints, real-time budget. Deferred to later phases.
- Non-goal: astronaut photography or NASA-sourced data as a dependency. Deployment is Argus imagery against a Sentinel-2 reference DB. EarthLoc data is bootstrap only.
