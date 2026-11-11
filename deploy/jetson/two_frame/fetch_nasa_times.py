"""Fetch capture time, ISS nadir point, altitude and lens for each photo in a pairs file,
from NASA's Gateway to Astronaut Photography (one page per frame). Saves after every
page and skips frames already fetched, so it can be stopped and resumed.

    python3 fetch_nasa_times.py pairs_heldout_sample40.json
"""
import html, json, os, re, subprocess, sys, time

path = sys.argv[1]
cache_path = "nasa_frames.json"
pairs = json.load(open(path))
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

def fetch(mission, frame):
    url = f"https://eol.jsc.nasa.gov/SearchPhotos/photo.pl?mission={mission}&roll=E&frame={frame}"
    r = subprocess.run(["curl", "-sS", "-m", "120", url], capture_output=True, text=True, errors="ignore")
    if r.returncode != 0 or "Time taken" not in r.stdout:
        return None
    t = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.stdout)))
    g = lambda pat: (re.search(pat, t).group(1).strip() if re.search(pat, t) else None)
    return {"date": g(r"Date taken (\S+)"), "time_gmt": g(r"Time taken (\S+) GMT"),
            "nadir": g(r"Spacecraft nadir point: (.+?) Photo"), "altitude_km": g(r"\((\d+)km\)"),
            "focal": g(r"Focal Length (\S+)")}

todo = [(p["mission"], p[f"frame_{s}"]) for p in pairs for s in ("a", "b")]
todo = [x for x in dict.fromkeys(todo) if f"{x[0]}-E-{x[1]}" not in cache]
print(f"{len(todo)} frames to fetch", flush=True)
for i, (m, f) in enumerate(todo):
    for attempt in range(3):
        rec = fetch(m, f)
        if rec:
            break
        time.sleep(10)
    cache[f"{m}-E-{f}"] = rec or {"error": "no page after 3 tries"}
    json.dump(cache, open(cache_path, "w"), indent=1)
    print(f"  {i + 1}/{len(todo)} {m}-E-{f}: {rec.get('time_gmt') if rec else 'failed'}", flush=True)
    time.sleep(1)
print("done", flush=True)
