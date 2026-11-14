# Real same-pass photo pairs

Finds pairs of real astronaut photos taken minutes apart on the same pass, as real two-frame input for OD. Commands: `../README.md` §5.8.

| File | Runs on | What it does |
|---|---|---|
| `find_pairs.py` | workstation | candidate pairs: same mission and date, ≤ 200 frames apart, footprints 200–1,000 km apart, both in held-out regions with a correct tile |
| `fetch_nasa_times.py` | Mac (internet) | capture times, nadir point and altitude from NASA's astronaut-photo site (resumable) |
| `check_same_pass.py` | Mac | marks a pair same-pass if the ISS nadir moved at 6–8.5 km/s between photos and the gap is under 25 min |
| `prepare_pair_set.py` | workstation | builds an `e2e_match.py` test folder from the confirmed pairs |
| `summarize_pairs.py` | any | pairs with a fix on both photos (and both under 50 km) |
| `*.json` | — | the sample of 40 pairs, NASA times, and the Jetson results |

Result: 38 of 40 sampled pairs are the same pass (12–147 s apart); 8 got fixes on both photos.
