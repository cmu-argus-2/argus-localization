"""Download a training-scoped subset of NASA GAPE's already-harvested mlcoord
metadata into a *writable* directory.

Background: /home/pvijayba/sentinel/expand_gape_queries.py already harvested
773,153 mlcoord records (full 4-corner footprints, AstroLoc's own labeling
pipeline run in production against GAPE's backlog -- see repo memory
gape_mlcoord_harvest) into gape_state/manifest.jsonl. That script's own
download phase is broken: its OUTPUT_DIR is /mnt/sdc1/astroloc/data/queries,
which is root-owned, so every write fails with EACCES (confirmed directly,
2,877/2,914 failures in its last logged run). This script reads that same
manifest but writes to REFERENCE_TRAIN_DIR/gape_queries instead, which is
writable, and applies a geographic + per-region cap (see astroloc/data/
regions.py) instead of downloading all 773k -- a full download was measured
at ~11.8 img/s with 32 concurrent workers (~18h for the full manifest, before
accounting for GAPE-side rate limiting), which does not fit a demo-scale
training budget.

Filename convention matches data_loading/earthloc_loader.py::parse_geotile_filename
exactly, so downloaded images drop straight into the existing GeoTile loader.
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from io import BytesIO

from astroloc.data.regions import TRAIN_QUERY_RADIUS_KM, TRAIN_REGIONS

MANIFEST_PATH = "/home/pvijayba/sentinel/gape_state/manifest.jsonl"
IMAGE_BASE = "https://eol.jsc.nasa.gov/DatabaseImages"
DEFAULT_OUTPUT_DIR = "/mnt/sdc1/astroloc/reference_db/astroloc_train/gape_queries"
OUT_PIXELS = 1024
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0
RATE_LIMIT_COOLDOWN_S = 60


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def assign_region(nadir_lat: float, nadir_lon: float) -> str | None:
    for name, (clat, clon) in TRAIN_REGIONS.items():
        if haversine_km(nadir_lat, nadir_lon, clat, clon) < TRAIN_QUERY_RADIUS_KM:
            return name
    return None


def select_records(manifest_path: str, per_region_cap: int, seed: int = 0) -> list[dict]:
    by_region: dict[str, list[dict]] = {name: [] for name in TRAIN_REGIONS}
    seen_keys = set()
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["mission"], rec["roll"], rec["frame"])
            if key in seen_keys:
                continue
            try:
                nlat, nlon = float(rec["nadir_lat"]), float(rec["nadir_lon"])
            except (KeyError, ValueError):
                continue
            region = assign_region(nlat, nlon)
            if region is None:
                continue
            seen_keys.add(key)
            by_region[region].append(rec)

    rng = random.Random(seed)
    selected = []
    for name, recs in by_region.items():
        rng.shuffle(recs)
        chosen = recs[:per_region_cap]
        print(f"  {name}: {len(chosen)}/{len(recs)} selected")
        selected.extend(chosen)
    return selected


def build_filename(rec: dict) -> str:
    lat1, lon1 = float(rec["ul_lat"]), float(rec["ul_lon"])
    lat2, lon2 = float(rec["ur_lat"]), float(rec["ur_lon"])
    lat3, lon3 = float(rec["lr_lat"]), float(rec["lr_lon"])
    lat4, lon4 = float(rec["ll_lat"]), float(rec["ll_lon"])

    width_km = haversine_km((lat1 + lat4) / 2, lon1, (lat2 + lat3) / 2, lon2)
    height_km = haversine_km(lat1, (lon1 + lon2) / 2, lat4, (lon3 + lon4) / 2)
    sq_km_area = int(round(width_km * height_km))

    image_id = f"{rec['mission']}-{rec['roll']}-{rec['frame']}"
    nadir_lat, nadir_lon = float(rec["nadir_lat"]), float(rec["nadir_lon"])
    orientation = float(rec["orientation"])

    return (
        f"@{lat1:.6f}@{lon1:.6f}@{lat2:.6f}@{lon2:.6f}@{lat3:.6f}@{lon3:.6f}"
        f"@{lat4:.6f}@{lon4:.6f}@{image_id}@{rec['pdate']}"
        f"@{nadir_lat:.1f}@{nadir_lon:.1f}@{sq_km_area}@{orientation:.1f}@.jpg"
    )


def fetch_image_bytes(directory: str, filename: str) -> bytes:
    url = f"{IMAGE_BASE}/{directory}/{filename}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT_S)
            if r.status_code == 429:
                raise RuntimeError("rate limited (429)")
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep_s = (
                RATE_LIMIT_COOLDOWN_S if "429" in str(e) else BACKOFF_BASE_S * (2**attempt)
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {last_err}")


def download_one(rec: dict, output_dir: str) -> str:
    fname = build_filename(rec)
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        return "skip_exists"
    img_bytes = fetch_image_bytes(rec["directory"], rec["filename"])
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = img.resize((OUT_PIXELS, OUT_PIXELS), Image.LANCZOS)
    tmp_path = out_path + f".part{os.getpid()}"
    img.save(tmp_path, "JPEG", quality=90)
    os.replace(tmp_path, out_path)
    return "fetched"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--per-region-cap", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="cap total downloads, for testing")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Selecting records within {TRAIN_QUERY_RADIUS_KM}km of each training region "
          f"(cap {args.per_region_cap}/region)...")
    records = select_records(MANIFEST_PATH, args.per_region_cap)
    if args.limit:
        records = records[: args.limit]
    total = len(records)
    print(f"Downloading {total} images with {args.workers} workers into {args.output_dir}...")

    counts = {"fetched": 0, "skip_exists": 0, "failed": 0}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, rec, args.output_dir): rec for rec in records}
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as e:  # noqa: BLE001
                counts["failed"] += 1
                print(f"  FAILED {rec['mission']}-{rec['roll']}-{rec['frame']}: {e}")
            done += 1
            if done % 1000 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
                print(f"  [{done}/{total}] {counts} rate={rate:.1f}/s eta={eta_min:.1f}min", flush=True)

    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
