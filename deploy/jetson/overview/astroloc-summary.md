# AstroLoc — Summary

**Berton, Stoken, Masone.** *AstroLoc: Robust Space to Ground Image Localizer.* arXiv:2502.07003 (2025).
No code released. `argus-localization` contains a reproduction — (this repo).

---

## The problem

Astronauts photograph Earth from the ISS by hand — up to 10k images a day, none geotagged. Finding where each one looks has been manual work, hours per photo.

Framed as **image retrieval**: embed the photo, search a database of geotagged satellite tiles, return the closest match. Coarse location in under a second.

## The pipeline

### First: what's a tile?

Two different sets of images are involved, and the whole method depends on the difference.

**The database.** Satellite imagery of the entire Earth, downloaded ahead of time and chopped into a grid of squares. Each square is a **tile**. Satellites know exactly where they were pointing, so **every tile comes with known lat/lon corners**. Google Maps, cut into a grid. Built once, before launch.

**The query.** The photo your camera just took. Location unknown — that's the problem being solved.

So the task is: *find the tile that shows the same piece of ground as my photo.* Once you know which tile, you know where the photo is looking, because the tile's coordinates were never in doubt.

### The two stages

```
query photo ──► RETRIEVE ──► ~100 candidate tiles ──► MATCH ──► the one that really fits
                                                                         │
                                                                         ▼
                                                        its corners = where the photo looks
```

**Retrieve.** Every tile was converted to a vector in advance. Convert the query to a vector too, then find the nearest ones. Fast — millions of tiles in milliseconds — but it only judges *what the image looks like*. Two different forests produce similar vectors.

**Match.** Take the query and one candidate tile, and find distinctive points in each: a bend in a coastline, a crater rim, a fork in a river. Check whether the points in the **query** correspond to points in the **tile**, and whether those correspondences fit a single consistent geometric transform. Slow, but it checks real structure rather than resemblance.

Two consequences:

- **R@100 matters more than R@1.** Everything in the shortlist gets re-checked by matching, so the question is whether the right tile is *anywhere* in the top 100 — not whether it ranked first.
- **The number of matched points is a confidence score.** A real match produces many consistent points; a wrong one produces few. Threshold it, and the system can answer **"I don't know"** instead of returning a confident wrong position — which matters on a satellite, where a wrong position corrupts everything downstream.

The paper's contribution is the **retrieve** stage. Matching is existing technique it builds on.

## The idea

Earlier models (EarthLoc) trained only on **satellite** imagery, but the images they must localize are **astronaut photos** — oblique, varied lighting, clouds, ISS hardware in frame. Different visual domain.

AstroLoc trains on the astronaut photos themselves. Two obstacles, two fixes:

**The labels were too weak to train on.** Manual labels give one point near the image center, with no extent — not enough to pair a photo with a tile. So they recovered full footprints automatically by matching each photo against candidate tiles (see below). Worked for 221k of 300k photos; failures were bad labels, heavy cloud, or horizon shots.

**Astronaut photos only cover part of Earth.** They cluster on glaciers, volcanoes, coastlines. To still use the full satellite database, they cluster it and then **sample those clusters in proportion to where astronauts actually point cameras** — deserts rarely, glaciers often. Training batches stay all-satellite; the photos only set the sampling weights.

Two losses, one per idea. Ablations show they're complementary — together they beat either alone by more than the sum.

## SuperPoint + LightGlue — the matching step

These are the **matching** half of the pipeline. They never produce a descriptor or a location themselves; they answer one question: *do these two images actually show the same ground?*

**SuperPoint** finds distinctive points in an image and describes each one — a coastline bend, a crater rim, a river fork. Roughly a thousand per image.

**LightGlue** takes the two sets of points and decides which correspond to which. It considers all points together rather than pairing them one at a time, so it can reject matches that are individually plausible but collectively inconsistent.

Then a geometric fit (`findHomography` with RANSAC) asks whether those correspondences fit **one consistent transform**. Points that agree are **inliers**; the rest are discarded.

### Why it matters

Retrieval only judges resemblance — two different forests produce similar vectors. Matching checks real geometry, which is why it catches those. But it is far too slow to run against millions of tiles, hence retrieve-first-then-match.

The by-product is the valuable part: **inlier count is a confidence score.** Real match, many consistent points; wrong match, few. Threshold it and the system can answer **"I don't know"** instead of a confident wrong position. `argus-localization` uses 30 inliers as that threshold, below which it returns `no_fix`.

### Where it feeds downstream

Matching is used twice in this project, for different purposes:

**In the paper, at training time.** Each weakly-labeled astronaut photo was matched against ~80 candidate tiles to recover its four corner coordinates. That is how 300k vague labels became 221k precise footprints — the training data the whole method depends on.

**In Argus, at inference time.** The matcher's inliers become **tie points** — each one pairing a pixel in the photo with a lat/lon on the ground. That chain continues:

```
matched keypoints ──► tie points (pixel ↔ lat/lon)
                            │
                            ├──► footprint: the photo's 4 corners
                            │
                            └──► pixel → bearing vector    ─┐
                                 lat/lon → ECI position    ─┴──► orbit solver
```

So matching is what ultimately supplies **position**. Retrieval alone narrows the search; the tie points are what the orbit determination solver actually consumes. See the FSW-Payload repo's `python_od`.

**One difference worth knowing:** the paper uses SuperPoint, but `argus-localization` uses **SIFT** with LightGlue (`LightGlue(features="sift")`). Same role, different keypoint detector — SIFT is classical and CPU-bound, which is why matching dominates runtime there at ~1-1.5 s/frame.

## Architecture

The model's only job: **turn a photo into one vector**, so photos of the same place land near each other. Retrieval is then just nearest-neighbour search over vectors.

```
photo ──► DINOv2 ──► SALAD ──► Linear ──► 2048-dim vector
          backbone   pooling   squeeze
```

**DINOv2 (backbone).** Splits the image into a grid of small patches and describes each one. Out comes ~256 patch descriptions plus a single summary of the whole image. Pretrained elsewhere, mostly frozen — only the last layers are tuned here.

**SALAD (pooling).** Squashes those ~256 descriptions into one fixed-length vector, so the answer doesn't change when a lake sits in the corner instead of the centre. It sorts patches into 64 learned buckets and sums each bucket.

The part worth knowing: SALAD includes a **"belongs nowhere" bucket**. Cloud, blank ocean and sensor noise get dumped there and thrown away before pooling — so uninformative pixels don't pollute the vector.

**Linear (squeeze).** Pooling produces 8448 numbers, too many to store for millions of tiles. One layer compresses to 2048 and normalizes, which makes similarity a plain dot product — the reason FAISS can search millions of these fast.

### Two sizes

| | Descriptor | Params |
|---|---|---|
| **AstroLoc** | 2048 | ~105M |
| **AstroLoc-tiny** | 512 | 27M |

Same design, smaller backbone and narrower output. Tiny exists to fit on a nanosatellite.

**Descriptor width sets database size, not just model speed.** At 512 dims a worldwide tile database is ~98 MB; at 2048 it's 4x that. On a satellite the database is usually the real constraint.

### For the ONNX export

- Input size must be divisible by 14, and the exported graph is locked to whatever size it was traced at.
- The Sinkhorn step inside SALAD is a fixed 3-iteration loop — no data-dependent branching, so it traces cleanly.
- Resize and normalize happen outside the model, so whatever runs the graph must do them.

## Results

Near-saturates the astronaut-photography benchmarks (>99% recall@100), roughly 35% better recall@1 than the previous best.

**The part relevant to Argus** — the "lost-in-space" test, using VINSat's own dataset as queries, with no fine-tuning:

| Model | R@1 | R@100 |
|---|---|---|
| EarthLoc | 3.3 | 27.6 |
| **AstroLoc** | **52.7** | **83.7** |
| AstroLoc-tiny | 36.7 | 74.5 |

EarthLoc collapses here; AstroLoc transfers. Shrinking to the tiny model costs ~16 points of R@1 — that's the price of fitting on a satellite.

**Failure modes:** forests (self-similar), coastlines with little land, and cloud cover. Cloud is the most common.

## What it means for Argus

- **It outputs a footprint, not a position.** The paper says to pair it with VINSat's state estimation. In Argus that bridge exists (FSW-Payload's orbit determination).
- **Worldwide coverage is affordable onboard.** At the resolution this task needs, a global tile database is ~98 MB.
- **Benchmark the small model against the small model.** 36.7 R@1, not the full model's 52.7.
- **Don't assume a good orbital prior.** Cosmic-ray bit flips can corrupt the timestamp, forcing a worldwide search.
- **Build the validation set to include forests, coasts, and clouds** rather than discovering them later.

---
