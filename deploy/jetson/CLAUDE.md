# Working rules for agents on Argus Jetson localization

Read `README.md` for commands, paths and setup, and `RUNBOOK.md` for results and their history. This file says how to work.

## Shared machines

The workstation and the Jetson are shared by the whole team. A mistake there affects everyone.

- **Show each command to your user before running it on the workstation or Jetson,** unless they've said to go ahead for that task. Read-only checks (`ls`, `nvidia-smi`, `tegrastats`, `df`) are fine to run.
- **Never** use sudo, reboot, change drivers or system packages, or kill or interrupt anyone else's process.
- **Workstation:**
  - Check `nvidia-smi` first and use a free GPU (`CUDA_VISIBLE_DEVICES=0`).
  - Work in your own home and venv.
  - The disk is nearly full, so remove large scratch output when done.
- **Jetson:**
  - Shared `argus` account, no internet. There's one working folder for everyone, `~/Desktop/astroloc/`. Read README §3.2 for what's already there before you create or change anything.
  - Don't move, overwrite or delete shared files (engines, tiles, tile features, `pydeps`, `e2e_alps_full/`, the top-level scripts). Your own work goes in `~/Desktop/astroloc/users/<name>/`, under your own run tags.
  - Install only into `~/Desktop/astroloc/pydeps` (`pip3 install --no-deps --target`).
  - Confirm nothing else is running (`ps`, `tegrastats`) before timing anything, since other workloads distort the numbers.
  - Don't overwrite files a teammate is using. Use your own run tags and output names (`--out`, the `run_match_full.sh` tag).
- **Never** ask for, store or paste passwords.
- Connect with the SSH aliases from README §3.1 (`argus-ws`, `argus-jetson`) and `-o BatchMode=yes`. If a password prompt appears, stop and ask your user to set up the key; never handle a password. Files go to the Jetson by streaming over SSH through the workstation (README §5.10). When you edit a script on your laptop, say which machine needs the new copy.

## How we run experiments

- **One change at a time,** with everything else fixed: same 200-photo test set, same top-k, same database.
- **Re-run the baseline on the same day** before comparing. Time, memory and power drift between days and after restarts.
- **Accuracy must be checked, not assumed.** Use `compare_topk.py` against the baseline and report fixes lost or gained and answers changed. A speed-up only counts if accuracy holds.
- **Log `tegrastats` with every timed run** (`run_match_full.sh` does this) and report time, peak board memory, average power and energy per photo.
- **Find where time goes before optimizing.** Use `profile_full.py` for per-step cost and `nsys` (`lg_bench.py`) for inside the GPU.
- **Verify new code against the old path** before timing it. Example: `test_batched_lightglue.py` checks batched matching against one tile at a time.
- **Record negative results too** (tested and rejected, with the numbers), so nobody repeats them.
- The 200 test photos are oblique, square astronaut photos. Say so when a result might not carry over to Argus frames (4608 × 2592, near-nadir).

## Writing results down

- Add every result to `RUNBOOK.md` as a dated table: what changed, the baseline next to it, and accuracy, time, memory and power. Add one line on whether it's adopted.
- Describe the system as it is now. Don't narrate earlier mistakes or beliefs ("we thought X, actually Y").
- Keep it plain and short: tables for numbers, short bullets, no filler. Define a term the first time it appears.
- **Team updates (all-hands):** 1–2 high-level lines per item, with few numbers. Sections: what I did / results / working / not working / blockers / plan.
- Keep each file's data in its own place: per-photo records in the `match_results_<tag>.json` files, logs next to them, summaries in the runbook.

## Before saying something is done

- Check the claim against the code or the output, not the README or memory.
- If you couldn't run something (the Jetson is unreachable, a GPU is down), say so plainly. Don't report it as done.
- Stale lines in the docs are bugs. When a result changes, update every place that quotes it (`README.md` §1, `RUNBOOK.md`).
