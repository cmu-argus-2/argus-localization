"""Test "stop at the first clearly-right tile" rules offline.

Reads a match_results JSON written by e2e_match.py --log-tiles (every candidate tile's inliers
and the fix error it would give) and replays each stop threshold: tiles are taken in retrieval
order, the best so far is kept, and matching stops at the first tile with >= N inliers.

  python3 simulate_stop.py RESULTS.json [--stops 0 50 75 100 150 200] [--min-inliers 30]
"""
import argparse
import json

import numpy as np


def replay(records, stop, min_inliers):
    fixes, errs, tiles = 0, [], []
    for r in records:
        best, n = None, 0
        for n, t in enumerate(r["tiles"], 1):
            if best is None or t["inliers"] > best["inliers"]:
                best = t
            if stop and t["inliers"] >= stop:
                break
        tiles.append(n)
        if best["inliers"] >= min_inliers and best["error_km"] is not None:
            fixes += 1
            errs.append(best["error_km"])
    errs = np.array(errs)
    return {
        "fixes": fixes,
        "fix_pct": 100 * fixes / len(records),
        "median_km": float(np.median(errs)) if fixes else float("nan"),
        "p90_km": float(np.percentile(errs, 90)) if fixes else float("nan"),
        "over_50km": int((errs > 50).sum()),
        "tiles_mean": float(np.mean(tiles)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--stops", type=int, nargs="+", default=[0, 40, 50, 75, 100, 150, 200])
    ap.add_argument("--min-inliers", type=int, default=30)
    args = ap.parse_args()
    records = json.load(open(args.results))["records"]
    print(f"{args.results}: {len(records)} photos")
    print(f"{'stop at':>8} {'fixes':>12} {'median km':>10} {'p90 km':>8} {'>50 km':>7} {'tiles/photo':>12}")
    for s in args.stops:
        o = replay(records, s, args.min_inliers)
        label = "off" if s == 0 else f">={s}"
        print(f"{label:>8} {o['fixes']:>5} ({o['fix_pct']:4.1f}%) {o['median_km']:>10.1f} {o['p90_km']:>8.1f} "
              f"{o['over_50km']:>7} {o['tiles_mean']:>12.2f}")


if __name__ == "__main__":
    main()
