"""Which same-pass pairs got a fix on both photos, from e2e_match.py's results.
    python3 summarize_pairs.py e2e_pairs/pairs.json e2e_pairs/pair_results.json
"""
import json, sys

pairs = json.load(open(sys.argv[1]))
res = {r["file"]: r for r in json.load(open(sys.argv[2]))["records"]}
both = [p for p in pairs if res[p["a"]]["fix"] and res[p["b"]]["fix"]]
good = [p for p in both if max(res[p["a"]]["match_error_km"], res[p["b"]]["match_error_km"]) < 50]
print(f"pairs: {len(pairs)} | both photos fixed: {len(both)} | both under 50 km: {len(good)}")
for p in both:
    a, b = res[p["a"]], res[p["b"]]
    print(f"  {p['region']:<12} {p['mission']}-E-{p['frame_a']} -> {p['frame_b']}  dt {p['dt_s']:4.0f} s  "
          f"errors {a['match_error_km']:5.1f} / {b['match_error_km']:5.1f} km  inliers {a['best_inliers']} / {b['best_inliers']}")
