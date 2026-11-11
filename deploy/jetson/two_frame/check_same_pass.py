"""Add NASA capture times to a pairs file and mark which pairs are really from the same pass.

Same pass = the ISS's nadir point moved at 6-8.5 km/s between the two photos (its
ground speed is ~7 km/s) and the gap is under 25 min. Reads the frame cache written by
fetch_nasa_times.py.
    python3 check_same_pass.py pairs_heldout_sample40.json
"""
import collections, json, math, re, sys
from datetime import datetime

path = sys.argv[1]
cache = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "nasa_frames.json"))
pairs = json.load(open(path))


def hav(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = p2 - p1, math.radians(d - b)
    return 6371 * 2 * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def nadir(s):
    m = re.match(r"([\d.]+)° ([NS]), ([\d.]+)° ([EW])", s or "")
    if not m:
        return None
    la, lo = float(m.group(1)), float(m.group(3))
    return (-la if m.group(2) == "S" else la, -lo if m.group(4) == "W" else lo)


for p in pairs:
    a, b = cache.get(f"{p['mission']}-E-{p['frame_a']}"), cache.get(f"{p['mission']}-E-{p['frame_b']}")
    p["nasa_a"], p["nasa_b"] = a, b
    if not a or not b or "error" in a or "error" in b or not a.get("time_gmt") or not b.get("time_gmt"):
        p["status"] = "no time"
        continue
    ta = datetime.strptime(a["date"] + " " + a["time_gmt"], "%Y.%m.%d %H:%M:%S")
    tb = datetime.strptime(b["date"] + " " + b["time_gmt"], "%Y.%m.%d %H:%M:%S")
    dt = (tb - ta).total_seconds()
    na, nb = nadir(a["nadir"]), nadir(b["nadir"])
    nd = hav(*na, *nb) if na and nb else None
    sp = nd / dt if nd is not None and dt > 0 else None
    p.update(dt_s=dt, nadir_km=nd, nadir_speed_kms=sp,
             status="same pass" if sp and 6.0 < sp < 8.5 and dt < 1500 else "not same pass")
json.dump(pairs, open(path, "w"), indent=1)
print(collections.Counter(p["status"] for p in pairs))
same = [p for p in pairs if p["status"] == "same pass"]
if same:
    dts = sorted(p["dt_s"] for p in same)
    print(f"same-pass gaps: {dts[0]:.0f}-{dts[-1]:.0f} s, median {dts[len(dts) // 2]:.0f} s")
