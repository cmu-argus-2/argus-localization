"""CPU-only inference profiling for the astroloc/ retriever: per-query latency,
RSS memory, and (when available) package power draw, logged to wandb alongside
the usual recall/coordinate metrics. Answers "what does this cost on a device
with no GPU" -- the Jetson-relevant number, unlike Table 3 of the AstroLoc
paper which profiles on GPU.

Reuses an already-built reference-db cache (rotation-TTA index included, see
database/reference_database.py's docstring) instead of rebuilding it -- only
query embedding runs on CPU, matching the real Jetson deployment shape where
the on-device model embeds one live frame at a time against a prebuilt index,
not the other way around.

Power draw needs read access to Intel RAPL's package energy counter, which is
root-only by default on this machine (checked: -r-------- root root on
/sys/class/powercap/intel-rapl:0/energy_uj). Without it this script still runs
and reports latency + RSS; power fields are reported as null with a printed
warning, not silently omitted or faked.
"""

import argparse
import json
import os
import sys
import threading
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import psutil

from database.reference_database import ReferenceDatabase, dedup_search
from index.faiss_index import FaissFlatIndex
from scripts.evaluate import (
    REGIONS,
    find_positive_tile_ids,
    footprint_bbox,
    haversine_km,
    load_image_array,
    load_scoped_queries,
)

from astroloc.models.dinov2_salad import DinoV2SaladRetriever

RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_RANGE_PATH = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"


class ResourceSampler:
    """Background thread sampling RSS and (if readable) package power at a
    fixed interval, for the duration of a `with` block. `samples` is a list
    of (elapsed_s, rss_mb, power_w_or_None) tuples.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float | None]] = []
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process()
        self.power_available = os.access(RAPL_ENERGY_PATH, os.R_OK)
        self._max_range_uj = None
        if self.power_available:
            try:
                with open(RAPL_MAX_RANGE_PATH) as f:
                    self._max_range_uj = int(f.read().strip())
            except OSError:
                pass

    def _read_energy_uj(self) -> int:
        with open(RAPL_ENERGY_PATH) as f:
            return int(f.read().strip())

    def _run(self, t_start: float):
        prev_energy = self._read_energy_uj() if self.power_available else None
        prev_t = time.time()
        while not self._stop.is_set():
            self._stop.wait(self.interval_s)
            now = time.time()
            rss_mb = self._process.memory_info().rss / (1024 * 1024)
            power_w = None
            if self.power_available:
                energy = self._read_energy_uj()
                delta_uj = energy - prev_energy
                if delta_uj < 0 and self._max_range_uj:  # counter wrapped
                    delta_uj += self._max_range_uj
                power_w = (delta_uj / 1e6) / max(now - prev_t, 1e-6)
                prev_energy, prev_t = energy, now
            self.samples.append((now - t_start, rss_mb, power_w))

    def __enter__(self):
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, args=(self._t0,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 2)

    def summary(self) -> dict:
        rss = [s[1] for s in self.samples]
        power = [s[2] for s in self.samples if s[2] is not None]
        return {
            "peak_rss_mb": max(rss) if rss else None,
            "mean_rss_mb": float(np.mean(rss)) if rss else None,
            "peak_power_w": max(power) if power else None,
            "mean_power_w": float(np.mean(power)) if power else None,
            "power_available": self.power_available,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--region", default="Alps", choices=list(REGIONS.keys()))
    ap.add_argument("--db-cache", required=True, help="prebuilt ReferenceDatabase dir (see astroloc/eval/evaluate.py --run-name output)")
    ap.add_argument("--queries-dir", default="/mnt/sdc1/astroloc/data/queries")
    ap.add_argument("--query-dist-km", type=float, default=2500)
    ap.add_argument("--iou-threshold", type=float, default=0.2)
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--sample-interval-s", type=float, default=0.5)
    ap.add_argument("--wandb-project", default="astroloc-demo")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch_threads_note = f"torch intra-op threads: {__import__('torch').get_num_threads()}"
    print(f"Loading retriever on CPU from {args.checkpoint} ...")
    retriever = DinoV2SaladRetriever.from_checkpoint(args.checkpoint, device="cpu")
    print(torch_threads_note)

    index = FaissFlatIndex(retriever.descriptor_dim)
    db = ReferenceDatabase(retriever, index)
    db = ReferenceDatabase.load(args.db_cache, retriever, index)
    print(f"Loaded reference db: {len(db.tiles)} tiles, {index._index.ntotal} rotation-augmented entries")

    center_lat, center_lon = REGIONS[args.region]
    queries = load_scoped_queries(args.queries_dir, center_lat, center_lon, args.query_dist_km)
    db_tiles = list(db.tiles.values())
    db_bboxes = np.array([footprint_bbox(t) for t in db_tiles])
    scored_queries = []
    for q in queries:
        if find_positive_tile_ids(q, db_tiles, db_bboxes, args.iou_threshold):
            scored_queries.append(q)
        if len(scored_queries) >= args.num_queries:
            break
    print(f"Profiling {len(scored_queries)} queries on CPU (region={args.region})...")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=f"cpu-profile-{args.region}", config=vars(args))

    sampler = ResourceSampler(interval_s=args.sample_interval_s)
    if not sampler.power_available:
        print(
            f"WARNING: {RAPL_ENERGY_PATH} not readable by this user -- power draw will be "
            f"reported as null. Fix: sudo chmod o+r {RAPL_ENERGY_PATH} (and the sibling "
            f"intel-rapl:0:0/energy_uj for DRAM domain, if present)."
        )

    latencies_ms = []
    top1_hits = 0
    errors_km = []
    with sampler:
        for query in scored_queries:
            image = load_image_array(query.image_path)
            t0 = time.perf_counter()
            descriptor = retriever.embed(image)
            top1_id, _ = dedup_search(index, descriptor, 1)[0]
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            positives = find_positive_tile_ids(query, db_tiles, db_bboxes, args.iou_threshold)
            if top1_id in positives:
                top1_hits += 1
            top1_tile = db.tiles[top1_id]
            pred_center = top1_tile.corners_latlon.mean(axis=0)
            gt_center = query.corners_latlon.mean(axis=0)
            errors_km.append(haversine_km(gt_center[0], gt_center[1], pred_center[0], pred_center[1]))

            if use_wandb and len(latencies_ms) % 10 == 0:
                import wandb

                wandb.log(
                    {
                        "query_idx": len(latencies_ms),
                        "latency_ms": latencies_ms[-1],
                        "rss_mb": sampler.samples[-1][1] if sampler.samples else None,
                        "power_w": sampler.samples[-1][2] if sampler.samples else None,
                    }
                )

    resource_summary = sampler.summary()
    result = {
        "region": args.region,
        "num_queries": len(scored_queries),
        "top1_correct_tile_rate": 100.0 * top1_hits / len(scored_queries) if scored_queries else 0.0,
        "median_coord_error_km": float(np.median(errors_km)) if errors_km else None,
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else None,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
        "descriptor_dim": retriever.descriptor_dim,
        **resource_summary,
    }
    print("\n=== CPU inference profile ===")
    print(json.dumps(result, indent=2))

    if use_wandb:
        import wandb

        wandb.log({f"summary/{k}": v for k, v in result.items() if isinstance(v, (int, float))})
        wandb.finish()


if __name__ == "__main__":
    main()
