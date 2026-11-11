# Start Here — Argus Localization in One Page

## 1. The pipeline

A CubeSat with no GPS photographs Earth and works out where it is.

```
photo ─► RETRIEVER ─► SEARCH ─► MATCHER ─► location ─► ORBIT SOLVER ─► orbit
         (ML)         database   (verify)   lat/lon     (FSW-Payload)
```

The **localization model = retriever + matcher**, same as the AstroLoc paper.

| Step | What it does |
|---|---|
| **Retriever** | Neural network (DINOv2 + SALAD + linear layer). Turns the photo into a fingerprint — a vector where photos of the same place come out similar |
| **Search** | Compares that fingerprint against pre-computed fingerprints of satellite map **tiles** with known coordinates. Returns the closest candidates |
| **Matcher** | SIFT + LightGlue. Checks each candidate actually lines up with the photo. Few matching points → "no fix" instead of a wrong answer |
| **Orbit solver** | Combines many locations over time with orbital physics into position and velocity |

**Photo vs tile:** a *photo* is the query (an astronaut or satellite-camera image). A *tile* is a satellite map image in the reference database.

### What each part is

| Part | Type | Off the shelf or trained? |
|---|---|---|
| **DINOv2-base** (retriever backbone) | Neural network (vision transformer): cuts the image into patches and describes each | Off the shelf (Meta, pretrained); its last layers are fine-tuned with small add-on weights (LoRA) |
| **SALAD** (retriever pooling) | Neural network layer: groups the patch descriptions into one vector, discarding clouds/ocean | Published design; trained as part of AstroLoc |
| **Linear layer** (retriever output) | One neural network layer: squeezes to 2,048 numbers | Trained as part of AstroLoc |
| → **AstroLoc retriever** = the three above | | **Trained by the team** (Priyanka's rebuild of the paper, checkpoint `faithful_lora_v2`; the paper released no code or weights) |
| **Search** | Plain math: dot product against stored tile fingerprints | No learning |
| **SIFT** | Classical algorithm: finds distinctive points (blob corners) and describes each with 128 numbers | Not learned (OpenCV) |
| **LightGlue** | Neural network (transformer with attention): decides which SIFT points in the photo and tile are the same place | Off the shelf: the authors' pretrained SIFT weights, not retrained by us |
| **Geometry check (MAGSAC)** | Classical algorithm: keeps the matches that fit one consistent transform; ≥ 30 = a fix | Not learned |
| **Orbit solver (OD)** | Physics + optimization (CasADi/IPOPT in FSW-Payload) | Not learned |

So two neural networks run on the Jetson: **the AstroLoc retriever (ours, trained)** and **LightGlue (off the shelf)**.

**"Full-size" vs "nano":** two sizes of the same AstroLoc retriever. Full-size: DINOv2-base, 2,048-number fingerprint, ~106M parameters, what we run. Nano (the paper's "AstroLoc-tiny"): smaller DINOv2, 512 numbers, 27M parameters, the backup.

## 2. EarthLoc vs AstroLoc

Two papers from the same group. AstroLoc (2025) is the newer, better one.

| | EarthLoc | AstroLoc |
|---|---|---|
| Retriever | ResNet50 + MixVPR, 4096 numbers | DINOv2 + SALAD, 2048 or 512 numbers |
| In our repo | authors' released weights, used as a baseline | **Priyanka's rebuild from the paper** — no code was released |
| Key idea | trained on satellite tiles | also trains on real astronaut photos, which look different from clean maps |

## 3. What runs on the Jetson

Both AstroLoc **retrievers** from `cmu-argus-2/argus-localization`, converted PyTorch → ONNX → TensorRT, plus the matcher.

| | Nano v2 | Full-size |
|---|---|---|
| Repo folder | `nano/` (DINOv2-small) | `astroloc/` (DINOv2-base) |
| Params / fingerprint | 27M / 512 numbers | 106.5M / 2048 numbers |
| Right tile ranked first | 34.5% | **68.5%** |
| Median error, retrieval only | 1,042 km | **79 km** |

- **Full-size is the flight candidate** — far more accurate, and it fits.
- **The full flow runs on the Jetson (Oct 5):** retriever + search + matcher in one process, with the same results as the workstation.
- **With the matcher:** a fix on 54% of astronaut photos (16 km median, footprint centre; 3 of 108 fixes > 50 km off) and 60% of Argus-like Landsat frames, where the matched points are within **0.7 km** median.
- **Cost on the 8 GB board:** **0.44 s and ~3.3 J per photo**, ~7.6 W while localizing, ~3.9 GB peak memory (3.3 s originally). Both networks run as TensorRT engines (retriever FP32, LightGlue FP16), and matching stops at the first tile with ≥ 30 matched points.
- **OD:** the Jetson's matched points into FSW-Payload's OD on a simulated pass give **~1.4 km** orbit error.
- Astronaut-photo numbers are from 200 photos around the Alps. Details in `../RUNBOOK.md`.

## 4. Data and deployment size

| | Size | Where |
|---|---|---|
| Tile fingerprint database | **249 MB** full-size, 62 MB nano (7,602 zoom-9 tiles; paper says ~98 MB for 12k with the tiny model) | deploys to the Jetson |
| Zoom-9 tile images for the matcher | 720 MB | deploys to the Jetson |
| All raw tile images | 414 GB (4 years 2018–2021 × zoom 8–12; folders named `{year}_{zoom}`, e.g. `2021_09`) | workstation `/mnt/sdc1/astroloc/data/database/` |
| Astronaut photos (queries) | ~17.8k | workstation `/mnt/sdc1/astroloc/data/queries/` |

The matcher works on each candidate tile's feature points, so besides the fingerprints the Jetson carries either the tile images (720 MB) or those points computed in advance (3.2 GB, and 2.5× faster per photo).

## 5. Training, testing and validation

- **Trained on:** astronaut photos (NASA GAPE + EarthLoc queries) paired with satellite tiles.
- **Tested on:** 6 held-out regions — Alps, Texas, Toshka Lakes, Amazon, Napa, Gobi.
- **Scored by:** recall (did the right tile come back?).
- *Source: training-code defaults; the wandb logs (`astroloc-demo`) confirm the actual runs.*

**Validation status:**
- **Argus-like imagery:** tested on 208 Landsat frames (`/mnt/sda2/geotiffs`): 60% fix, matched points within 0.7 km. Northern zones with snow/winter imagery fix much less often.
- **Coordinate error in km:** measured per matched point (above). Priyanka and Pedro are modelling the error for OD.
- **OD:** localization → FSW-Payload OD runs end to end on a simulated pass (~1.4 km).

## 6. The three repos

| Repo | Role |
|---|---|
| **argus-localization** | AstroLoc + EarthLoc retrieval research — the pipeline above. Workstation |
| **FSW-Payload** | Flight software on the Jetson. Runs detection models + the orbit solver |
| **Vision_Payload** | The older VINSat-style landmark-detection method, which also feeds OD |

## 7. Details

| Need | File |
|---|---|
| Jetson commands and results | `../RUNBOOK.md`; setup and machines in `../README.md` |
| AstroLoc paper | arXiv:2502.07003; summary `astroloc-summary.md` |
