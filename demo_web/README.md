# Argus · Where on Earth? (public web demo)

An interactive, static web demo of the retrieve-then-match localizer. A viewer
clicks the globe, gets the nearest real ISS photo, presses **Localize**, and
watches a replay of the actual pipeline run on that photo:

1. DINOv2+SALAD embed
2. FAISS top-15
3. SIFT-LightGlue on each candidate
4. Georeference

It ends with the predicted lat/lon and the error vs. ground truth. A
"Beat the AI" mode lets the viewer guess first.

## What's real, what's hosted where

- **Results:** every similarity, inlier count, correspondence, footprint,
  error, and timing comes from `export_demo_data.py`. That script runs the
  `faithful_lora_v2` checkpoint plus the matcher on the lab GPU, doing the
  same steps as `core/pipeline.py::LocalizationPipeline.localize`. The site
  only animates these numbers. The one decorative element is the
  descriptor-bar wiggle during the embed step.
- **No images are re-hosted.** The published bundle is JSON only, about 2 MB.
  - **Astronaut photos** load live from NASA GAPE
    (`eol.jsc.nasa.gov/DatabaseImages/ESC/small/...`). EarthLoc query images
    are GAPE's full frame resized to a square (verified pixel-identical), so
    the normalized correspondences map onto GAPE's image unchanged.
  - **Reference tiles** are rebuilt in the browser from EOx's public
    Sentinel-2 cloudless 2021 WMTS. Tile id `ZZ_YYYY_XXXX@2021` is exactly a
    4x4 block of web-mercator tiles at zoom `ZZ` starting at `(x, y)`. This
    was verified against all 4,695 exported candidates' corners, with
    correlation 0.993 against the on-disk tile.
- **The lab machine is not needed at view time.** It is only used to
  regenerate the bundle.

## Regenerate the data

```bash
python demo_web/export_demo_data.py --device cuda:0 --n-per-region 80
```

This takes about 12 minutes on one GPU and writes `demo_web/site/data/`:

- `manifest.json`: photo list and headline stats
- `q/<id>.json`: the full per-photo pipeline trace

The gallery keeps every fix and 30% of the no-fixes (`--nofix-keep-frac`).
The headline stats on the site come from the full unfiltered
`output/error_dataset/faithful_lora_v2.jsonl` run.

## Preview locally

```bash
cd demo_web/site && python -m http.server 8000   # open http://localhost:8000
```

## Publish (GitHub Pages)

`demo_web/site/` is self-contained. Push its contents to the root of any
public repo, then enable Pages from that branch.
