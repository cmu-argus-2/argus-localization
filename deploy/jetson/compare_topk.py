"""Compare full-flow runs side by side: accuracy, time, memory, power.

Reads match_results_<tag>.json and tegrastats_match_<tag>.log from the e2e folder.
The first tag is the baseline the others are compared against.
    python3 compare_topk.py --data e2e_alps_full --ks 15 10 5          # tags k15 k10 k5
    python3 compare_topk.py --data e2e_alps_full --tags k10 k10_mp k10_kp512
"""

import argparse
import json
import os
import re

import numpy as np


def tegra(path):
    ram, pw = [], []
    for line in open(path):
        ram.append(int(re.search(r"RAM (\d+)/", line).group(1)))
        pw.append(int(re.search(r"VDD_IN (\d+)mW", line).group(1)))
    run = pw[12:-4] or pw
    return max(ram), max(pw) / 1e3, sum(run) / len(run) / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="e2e_alps_full")
    ap.add_argument("--ks", type=int, nargs="+", default=[15, 10, 5])
    ap.add_argument("--tags", nargs="+", default=None, help="run tags; overrides --ks")
    args = ap.parse_args()
    tags = args.tags or [f"k{k}" for k in args.ks]

    rows, recs = {}, {}
    for k in tags:
        d = json.load(open(os.path.join(args.data, f"match_results_{k}.json")))
        s, r = d["summary"], d["records"]
        recs[k] = r
        a, u = s["all"], s["excluded_from_training"]
        ram, pk, avg = tegra(os.path.join(args.data, f"tegrastats_match_{k}.log"))
        sec = s["seconds_per_photo_median"]
        rows[k] = [
            ("correct tile among those checked", f"{a['correct_tile_in_topk_pct']}%"),
            ("fix rate", f"{a['fix_rate_pct']}% ({a.get('fixes', 0)})"),
            ("fixes on a correct tile", f"{a.get('fix_on_correct_tile_pct', '-')}%"),
            ("fix error median / p90", f"{a.get('fix_median_error_km')} / {a.get('fix_p90_error_km')} km"),
            ("fixes > 50 km off", str(a.get("fixes_over_50km"))),
            ("169 unseen: fix rate / median", f"{u['fix_rate_pct']}% / {u.get('fix_median_error_km')} km"),
            ("time per photo (median)", f"{sec} s"),
            ("board memory peak", f"{ram} MB"),
            ("power peak / average", f"{pk:.1f} / {avg:.1f} W"),
            ("energy per photo", f"~{avg * sec:.0f} J"),
        ]
    print(f"{'':34s}" + "".join(f"{k:>22s}" for k in tags))
    for i, (name, _) in enumerate(rows[tags[0]]):
        print(f"{name:34s}" + "".join(f"{rows[k][i][1]:>22s}" for k in tags))

    base = tags[0]
    ranks = [r.get("best_rank") for r in recs[base] if r["fix"] and r.get("best_rank") is not None]
    if ranks:
        print(f"\nwinning tile's search rank, {base} fixes: " +
              ", ".join(f"#{lo + 1}-{hi}: {sum(lo <= x < hi for x in ranks)}" for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 15))))
    for k in tags:
        if k == base:
            continue
        a, b = recs[base], recs[k]
        lost = sum(x["fix"] and not y["fix"] for x, y in zip(a, b))
        gained = sum(y["fix"] and not x["fix"] for x, y in zip(a, b))
        changed = sum(x["fix"] and y["fix"] and abs(x["match_error_km"] - y["match_error_km"]) > 1e-6 for x, y in zip(a, b))
        print(f"{k} vs {base}: fixes lost {lost}, gained {gained}, same photo but different answer {changed}")


if __name__ == "__main__":
    main()
