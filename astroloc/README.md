# astroloc

A demo-scale reproduction of **AstroLoc** ("AstroLoc: Robust Space to Ground
Image Localizer", arXiv:2502.07003, Berton/Stoken/Masone), replicating its
retrieval architecture and training losses up through recall@k and a
retrieval-only coordinate estimate. It does not (yet) include SIFT-LightGlue
matching/homography -- that stage already exists in the main pipeline
(`core/pipeline.py`) and is a separate, later integration.

See repo memory `astroloc_target_architecture` and `gape_mlcoord_harvest` for
the fuller research context this was built from.

## What this replicates from the paper

- **Architecture**: DINOv2-base backbone + SALAD (Sinkhorn optimal-transport
  aggregation), 8448-dim raw descriptor, linear-reduced to 2048-dim.
  `models/dinov2_salad.py`, vendored SALAD code in `third_party/salad/`
  (official release, github.com/serizba/salad), initialized from their
  released GSV-Cities-pretrained checkpoint rather than trained from scratch
  (a fine-tune, not a from-scratch reproduction -- see "Simplifications").
- **Two losses**: `L_pairs` (pairwise attraction/repulsion between a query and
  its matched satellite tile) and `L_MUM` (multi-similarity loss over
  k-means-clustered satellite tiles, k=50). Both built on one shared
  Multi-Similarity loss primitive (`losses/multi_similarity.py`), matching
  the paper's own hyperparameters (alpha=1, beta=50). `losses/pairwise.py`,
  `losses/mum.py`, `training/cluster.py`.
- **No re-labeling step**: the paper's own contribution of turning 300k
  weakly-labeled astronaut photos into 221k fully-footprinted ones (via
  SuperPoint+LightGlue+EarthMatch) is not reproduced, because NASA GAPE's
  `mlcoord` table already publishes that pipeline's output at far larger
  scale (846k+ records) -- this was confirmed directly, see memory
  `gape_mlcoord_harvest`. `data/gape_download.py` pulls a training-scoped
  subset of it directly.
- **Positive pairing**: since every query already carries a full 4-corner
  footprint (either from mlcoord or the original EarthLoc queries), a
  positive (query, tile) pair is just a geometric IoU check
  (`data/pairing.py`, reusing `scripts/evaluate.py`'s IoU machinery), not a
  feature-matching problem.
- **Reference/satellite tiles**: reuses the existing rsynced EarthLoc
  database mirror (real Sentinel-2-derived cloud-free composites, already on
  disk) instead of pulling a new EOxCloudless database from scratch.
- **Eval**: recall@1/5/10/100 (the paper's own k-values) on the same 6
  EarthLoc/AstroLoc benchmark regions (Alps, Texas, Toshka Lakes, Amazon,
  Napa, Gobi), reusing `scripts/evaluate.py`'s region scoping and IoU ground
  truth unchanged -- only the retriever is swapped in. `eval/evaluate.py`.

## Simplifications vs. the paper (demo-scale, not a full reproduction)

- **Fine-tune, not from-scratch training.** Starts from SALAD's own
  GSV-Cities-pretrained weights (last 4 DINOv2 blocks + the SALAD head + a
  new reduction layer are trainable, ~47M of 105M total params) rather than
  the paper's own training regime, since that converges far faster than
  training the retrieval geometry from nothing.
- **Training data is a small, hand-picked geographic slice, not worldwide.**
  Six "coastal training regions" (`data/regions.py::TRAIN_REGIONS`:
  Philippines, Australia east coast, Aegean, Japan, Gulf of Guinea,
  Caribbean) stand in for the paper's full worldwide (minus held-out) scope.
  This was a direct consequence of measuring real GAPE download throughput
  (~11.8-16 img/s with 32-40 workers -- the full 846k-record manifest would
  take roughly a day; even the user's own suggested 400k-image fallback was
  not realistically downloadable inside a single-session budget) and DINOv2
  embedding cost being much higher than the existing EarthLoc pipeline's
  ResNet50. Some of these training centers are only ~1500-3000km from an
  eval region center, so the zero-shot claim on the 6 benchmark regions is
  reasonable but not airtight -- see `data/regions.py`'s own docstring.
- **Clusters computed once, not recomputed periodically.** The paper
  recomputes k-means clusters every 5000 iterations as the embedding space
  moves; this reproduction clusters once up front from the pretrained
  model's embeddings, mainly because the fine-tune converges quickly enough
  from a strong initialization that cluster drift matters less.
- **One shared batch for both losses, not separate pairs/quadruplet
  batches.** Each training step computes both `L_pairs` and `L_MUM` on the
  same uniformly-shuffled batch of positive pairs, rather than the paper's
  separate 24-pairs/24-quadruplets batch composition.
- **No hybrid/YOLO fallback, no OD integration.** Out of scope here exactly
  as it is for the rest of this repo (see main README's phased plan).

## Running it

```bash
# 1. Pull a training-scoped subset of the already-harvested GAPE mlcoord
#    metadata (see repo memory gape_mlcoord_harvest for why the original
#    download location doesn't work). Takes roughly an hour at ~75k images.
python -m astroloc.data.gape_download --per-region-cap 15000 --workers 40

# 2. Train (builds+caches positive pairs and clusters on first run).
python -m astroloc.training.train --tile-cap 20000 --epochs 40

# 3. Evaluate: recall@k + retrieval-only coordinate estimate on the 6
#    EarthLoc benchmark regions, both for the off-the-shelf pretrained
#    checkpoint (baseline) and the fine-tuned one, logged to wandb.
python -m astroloc.eval.evaluate --run-name baseline
python -m astroloc.eval.evaluate --run-name finetuned \
    --checkpoint /mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints/final.pt
```

`data/gape_download.py`, training pair/cluster caches, checkpoints, and eval
caches all live under `/mnt/sdc1/astroloc/reference_db/astroloc_train/`
(writable; the original `/mnt/sdc1/astroloc/data/{queries,database}` are
root-owned read-only mirrors this repo already used). wandb project:
`astroloc-demo` (entity `ARGUSVision`).
