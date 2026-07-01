# Argus Localization: Retrieval plus Matching Design

**Status:** Draft / hypothesis. Nothing here is validated yet. The point of this doc is to describe the architecture, say why we think it helps, and list the exact measurements that would prove or kill it.

**Author:** Priyanka
**Audience:** Argus team (GNC / payload)

**Update (2026-07-01, confirmed with Priyanka):** every "Sentinel-2" mention below for the Phase 1 reference DB is superseded. The actual available imagery on this machine is Landsat-8 (`/mnt/sda2/geotiffs/`, 16 MGRS zones, `l8_` filename prefix, B4/B3/B2 bands, 175m/pixel, ~500 temporal composites per zone). Read "Sentinel-2" as "Landsat-8" throughout until this doc is revised.

---

## 1. What we are trying to do

Localization means working out where on Earth a frame was taken: the ground location and extent of what the camera saw. Argus needs this to feed orbit determination.

This doc proposes changing how that location is found. It does not touch orbit determination or the optimizer math. It changes the part that turns an image into ground measurements.

## 2. The current pipeline

Today, per frame, the pipeline does roughly this:

1. RCNet looks at the frame and classifies which region it is.
2. That region picks which YOLO network (or set of YOLO networks) runs.
3. The YOLO networks detect known landmarks.
4. The landmarks, which have known world coordinates, go into batch optimization.
5. Batch opt feeds orbit determination.

This works, but it has two weak spots, and both come from the same root cause: RCNet and YOLO are both closed-set. They only know the classes they were trained on.

- **RCNet mis-routes novel frames.** If a frame is oblique, or just unlike anything in training, RCNet still picks a region, often with false confidence. There is no "I do not recognize this" option. A wrong region sends the frame to the wrong YOLO networks, which then hunt for landmarks that are not there. The error slips downstream silently.
- **YOLO only finds trained landmark types.** Anywhere without those specific landmarks produces no measurements, even if the scene is full of usable structure.

For the oblique and near-oblique frames we expect on Argus, both weak spots get worse.

## 3. The proposed pipeline

Replace the closed-set region step (and optionally the landmark step) with an open-set retrieval plus matching pipeline. The ideas come from two papers in the astronaut photography localization line of work, EarthLoc (retrieval) and EarthMatch (matching), adapted to Argus.

Per frame:

1. **Retrieve.** A small visual model turns the frame into one descriptor vector. We compare that vector against a precomputed database of reference satellite tiles whose locations we know. This returns a shortlist of the top-k most similar tiles.
2. **Match.** A feature matcher (SIFT-LightGlue) takes the frame and each shortlisted tile and tries to line them up by their actual image features. If they overlap, it returns a homography: a dense mapping from frame pixels to ground coordinates. If they do not overlap, it returns too few matched points, and we reject that candidate.
3. **Feed batch opt.** The matched correspondences are points in the image with known world coordinates. That is exactly what batch opt wants. The matcher can feed batch opt directly, with many more correspondences than a handful of YOLO landmarks, anywhere the scene has texture.

So retrieval answers "roughly where is this," and matching answers "exactly which pixels map to which ground points, and do I trust it."

## 4. Why this is worth doing

The honest headline reason is robustness, not speed. Speed is a maybe (see section 5). Robustness is the real argument.

- **Open-set behavior.** Retrieval similarity and matcher inlier count are both numbers we can threshold. A weak match means "do not trust this," so we reject or flag the frame instead of feeding garbage measurements into orbit determination. The closed-set pipeline cannot do this.
- **Generalizes past training.** Matching works on scenes the model never saw, as long as there is texture to lock onto. We are not limited to trained landmark classes or trained regions.
- **Built for obliques.** This whole method family was designed for handheld, oblique, rotated astronaut photography, which is a harder version of what Argus will see. Our near-nadir case is the easier end of what it already handles.
- **Richer measurements.** A homography gives many correspondences, not just the few landmarks YOLO was trained to find. That is a denser measurement model for batch opt.

A useful way to see it: the matcher can replace both RCNet and the YOLO landmark set, because it does both jobs (find the place, produce measurements) with one open-set mechanism instead of two closed-set ones.

## 5. Why it might also be cost-competitive (not assumed, tested)

The current path fires RCNet plus a set of YOLO networks every frame. That is a heavy front end. So the bar the new path has to clear is not one cheap detector, it is several detector passes.

New path cost per frame:
- one small descriptor pass
- a nearest-neighbor search over precomputed tile descriptors
- the matcher run over the top-k candidates, where the database-side features are precomputed offline so only the query side is processed at runtime

Whether this beats the current path depends on two numbers: how many YOLO passes are in the current set, and how big k is. We already know the current pipeline fires a set, not one network, so the comparison is live. It is not obviously cheaper and not obviously more expensive. It has to be measured on the Orin.

**Do not sell this internally as "it is cheaper."** Sell it as "it is open-set and feeds batch opt a richer measurement model." Treat cost as something to verify.

## 6. The DINOv2 question (key design decision)

EarthLoc and AstroLoc use a DINOv2-base backbone for retrieval. DINOv2-base is too heavy for our hardware, both in compute and in the memory footprint of running it alongside everything else on the Orin.

The instinct is to worry that a smaller retriever will be much worse, because DINOv2's quality comes from heavy pretraining. That worry is correct only in a retrieval-only system, where the top-1 result is the final answer and you need high R@1.

In our design, the top-1 is not the final answer. The matcher verifies and ranks the shortlist by real geometric overlap, which is a stronger test than descriptor similarity. So the retriever does not need to rank the right tile first. It only needs to keep the right tile somewhere in the top-k it hands the matcher.

That changes the job from R@1 (hard) to recall@k (much easier). And recall-into-a-shortlist is exactly where small models are already good enough on textured satellite imagery.

How we make a small retriever good enough at recall@k:

- **Distill from DINOv2.** Train the small model to copy DINOv2's embedding geometry, specifically the neighbor relationships (which tiles are near which), rather than training from scratch on labels. We have the teacher weights already.
- **Train narrow.** DINOv2 is strong because it is general. We do not need general. We need to be good on near-nadir satellite tiles of Earth at our scale and bands. A small model trained only on that distribution can match a general giant inside that distribution.
- **Use augmentations for the invariances we actually need.** Rotation augmentation for orientation, photometric jitter for lighting and atmosphere, and multi-temporal sampling (same place across years and seasons) for seasonal change. These are the specific nuisance axes Argus sees, and each maps to a concrete augmentation. That is far less than DINOv2 had to learn.
- **Keep a strong aggregation head.** How patch features get pooled into the final descriptor matters a lot. Shrink the backbone, but keep a good head (SALAD or GeM).

## 7. Pose priors (deliberately kept secondary)

Argus knows roughly where it is and where it is pointed, far better than a handheld astronaut camera. We could generate candidate tiles geometrically from orbit and attitude and skip visual retrieval for many frames.

We are choosing not to make this the primary path, because the open-set robustness of visual matching is the main thing we want, and a pose-only candidate is only as good as our attitude knowledge. Pose priors are kept as an optional speedup and a sanity check on retrieval, not the main mechanism. The matcher's accept/reject stays as the safety check regardless.

## 8. Open risks (the part that decides if this is real)

Every one of these is currently untested. Each is a place the design could fail or need rework.

1. **Retriever recall.** The small model might drop the correct tile out of the top-k entirely. In retrieval-only that is a ranking miss; here it is fatal, because the matcher never sees a tile that is not in the shortlist. **Track recall@k, not R@1.** A model with mediocre R@1 but strong recall@k is a success here.
2. **Matcher cost and timing variance.** The matcher is iterative, so per-frame time varies with the scene. A flight loop budgets against worst case, not median. We need a hard iteration cap and a fail-fast on low inliers so one bad frame cannot blow the time budget.
3. **Textureless scenes.** The matcher needs texture. Ocean, uniform desert, sea ice, dense cloud, and sun-glint give it almost nothing, and the homography collapses. This is the opposite failure mode from a trained landmark detector, which can be built to find the few reliable features in those scenes. If Argus spends a lot of its orbit over open ocean, this matters a lot. We may need YOLO landmark detection as a fallback for low-inlier frames (a hybrid), not a full replacement.
4. **Scale alignment.** Retrieval only works if the query frame and the database tiles are at a comparable ground scale. EarthLoc's queries cover about 25,000 sq km each, which is nothing like an Argus frame. So we cannot reuse EarthLoc's database wholesale. The reference database has to be rebuilt at Argus's actual scale. This is the single most common way the numbers could look bad for a reason that is not the method's fault.
5. **Output adapter.** The matcher output (homography to world-coordinate correspondences) has to be put into whatever measurement format batch opt consumes. If we keep YOLO routing as a fallback, we also need a tile-to-region lookup. Both are small but real.

## 9. Phased plan

A phase only starts when the previous one gives a usable result. This keeps us from building the small-model and matcher work before we know retrieval works at our scale.

- **Phase 0, baseline harness.** Run the released EarthLoc model on the rsync data. Confirm the eval harness works and get baseline recall numbers on the six EarthLoc eval regions. This validates plumbing, not Argus performance.
- **Phase 1, scale alignment.** Build a small reference database at Argus scale from Sentinel-2, using our existing MGRS tiling. Confirm retrieval returns sane candidates when query and database scales match. This is the make-or-break check.
- **Phase 2, small retriever.** Distill the small descriptor from DINOv2, train narrow with augmentations, and measure recall@k against the DINOv2 retriever. Decide the smallest model that holds recall@k at our chosen k.
- **Phase 3, matcher integration.** Run the matcher over the top-k, produce correspondences, and feed batch opt. Measure end-to-end localization quality.
- **Phase 4, hardware and scene tests.** Benchmark on the Orin: median and worst-case frame time, current path versus new path. Stratify quality by scene type, especially textureless frames. Decide pure matching versus matcher-primary-with-YOLO-fallback.

## 10. Benchmarks that resolve the risks

Three concrete measurements settle the whole design. None of them needs a rebuild to start.

1. **Timing, on the Orin.** Wall-clock for one current-path frame (RCNet plus the full YOLO set) versus one new-path frame (small descriptor plus kNN plus a lean matcher over k candidates, with database keypoints precomputed). Record median and worst case.
2. **Recall, on EarthLoc queries.** Does the small retriever keep the correct tile in the top-k? Report recall@k at the chosen k, not R@1.
3. **Scene-stratified quality.** Across scene types, compare matcher correspondence count against YOLO landmark count. Find exactly where matching wins and where it falls off, especially over textureless terrain.

---

## 11. Model suggestions and specifics

These are starting points, not final choices. Tune against the benchmarks above.

### Retriever (the shortlist stage)

- **Backbone:** a compact CNN you can quantize and run on the Orin. EfficientNet-Lite or MobileNetV3 are good first picks given your EfficientNet experience. Avoid a ViT here; the whole point is to drop the heavy transformer.
- **Aggregation head:** keep GeM as the cheap strong default. SALAD is the AstroLoc choice and is fine, but it produces a high-dimensional descriptor (about 8448 before reduction) that you then reduce with a linear layer. GeM keeps things lighter. Keep the head strong even though the backbone is small.
- **Descriptor dimension:** start at 512, or smaller. This directly sets database memory and kNN cost, so keep it modest.
- **Training:** distill from the DINOv2-base plus SALAD weights you already have. Use relational distillation (match the teacher's pairwise similarity structure within a batch) rather than copying raw embeddings, because recall@k depends on neighbor relationships, not absolute vectors. Add a domain-specific fine-tune on your Sentinel tiles.
- **Augmentations:** rotation (also apply at test time, like AstroLoc), photometric jitter, and multi-temporal pairs (the same place across 2018 to 2021 is already in the EarthLoc database).
- **Deployment:** quantize to INT8.

### Matcher (the precision stage)

- **Default:** SIFT-LightGlue, the EarthMatch choice. Well validated for this exact task.
- **Lean config for Argus:** near-nadir frames against scale-matched tiles need far less than EarthMatch's oblique defaults. Start around `max_num_keypoints = 1024` (down from 2048) and `img_size = 512`, then push lower if timing needs it.
- **Latency bounding:** hard cap on iterations (start at 3) and a minimum-inlier threshold for accept. Below the threshold, reject the frame (or hand it to the YOLO fallback) instead of iterating.
- **Precompute the database side:** compute and store every reference tile's SIFT keypoints offline, so runtime only extracts keypoints for the query.
- **Alternatives:** the image-matching-models repo exposes other matchers if SIFT-LightGlue is too slow or too weak on your scenes. Worth a sweep only if the default underperforms.

### Nearest-neighbor search and database

- **Library:** FAISS. Flat index is fine for a small database. Move to IVF-PQ only if the database grows large enough that RAM becomes the limit (it compresses stored vectors a lot at a controlled recall cost).
- **Build the reference database at Argus scale** from Sentinel-2 L2A (clean open licensing, already surface-reflectance corrected), reusing your MGRS tiling on the 4090s. Do not inherit EarthLoc's coarse tiles.
- **Store per tile:** the descriptor (for retrieval) and the precomputed SIFT keypoints (for matching).

### Starting hyperparameters to put in the first run

- shortlist size `k`: start at 10 to 20, measure recall@k, then trade down toward 5 if matcher cost dominates.
- descriptor dim: 512.
- matcher: `sift-lg`, `max_num_keypoints = 1024`, `img_size = 512`, max iterations 3, minimum inliers threshold set from your Phase 3 data.
- retriever: INT8, GeM head, distilled then domain-fine-tuned.

### Integration shape

- Start with **matcher-primary, YOLO-fallback**: run retrieval plus matching first, and only fall back to the existing YOLO set on low-inlier frames. This keeps the open-set win and the cost win where they apply, while covering the textureless blind spot. Collapse to pure matching later only if the scene tests say the fallback is rarely needed.
- Write the **homography-to-measurement adapter** once: matched correspondences with world coordinates into batch opt's measurement format.
