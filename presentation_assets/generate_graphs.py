"""Generates the overall (not per-region) result graphs for the presentation,
reading directly from the saved eval_cache/results_*.json files so numbers
stay accurate if a run gets re-evaluated (e.g. nano v2 finishing later).

Run: python presentation_assets/generate_graphs.py
Outputs PNGs into this same folder.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ASTROLOC_CACHE = "/mnt/sdc1/astroloc/reference_db/astroloc_train/eval_cache"
NANO_CACHE = "/mnt/sdc1/astroloc/reference_db/nano_train/eval_cache"

COLORS = {
    "baseline": "#94a3b8",
    "finetuned": "#60a5fa",
    "lora": "#2563eb",
    "nano": "#16a34a",
}


def load_avg(path):
    d = json.load(open(path))
    r = {
        "1": sum(v["recalls"]["1"] for v in d.values()) / len(d),
        "5": sum(v["recalls"]["5"] for v in d.values()) / len(d),
        "10": sum(v["recalls"]["10"] for v in d.values()) / len(d),
        "100": sum(v["recalls"]["100"] for v in d.values()) / len(d),
    }
    errs = [v["coords"]["median_coord_error_km"] for v in d.values() if v["coords"]["median_coord_error_km"] is not None]
    median_err = sum(errs) / len(errs) if errs else None
    return r, median_err, len(d)


def load_avg_merged(paths):
    combined = {}
    for p in paths:
        combined.update(json.load(open(p)))
    r = {
        "1": sum(v["recalls"]["1"] for v in combined.values()) / len(combined),
        "5": sum(v["recalls"]["5"] for v in combined.values()) / len(combined),
        "10": sum(v["recalls"]["10"] for v in combined.values()) / len(combined),
        "100": sum(v["recalls"]["100"] for v in combined.values()) / len(combined),
    }
    errs = [v["coords"]["median_coord_error_km"] for v in combined.values()]
    return r, sum(errs) / len(errs), len(combined)


baseline_r, baseline_err, _ = load_avg(f"{ASTROLOC_CACHE}/results_baseline.json")
finetuned_r, finetuned_err, _ = load_avg(f"{ASTROLOC_CACHE}/results_finetuned.json")
lora_r, lora_err, _ = load_avg_merged([
    f"{ASTROLOC_CACHE}/results_lora_finetuned_a.json",
    f"{ASTROLOC_CACHE}/results_lora_finetuned_b.json",
])
nano_r, nano_err, _ = load_avg(f"{NANO_CACHE}/results_nano-dinov2s-dynamic-512d.json")
_nano_v2_path = f"{NANO_CACHE}/results_nano-v2-multizoom-lora-dynamic.json"
nano_v2_r, nano_v2_err, _ = load_avg(_nano_v2_path) if os.path.exists(_nano_v2_path) else (None, None, None)

# Per-region model registry, used by the per-region graphs below. Each entry is
# (label, color, list of result-json paths to merge, region-name-remap).
# nano v2's path is a forward reference -- guarded by os.path.exists, so once
# nano/eval.py is run for it (--run-name nano-v2-multizoom-lora-dynamic) this
# script picks it up automatically on the next run, no edits needed.
PER_REGION_MODELS = [
    ("Baseline (pretrained)", COLORS["baseline"], [f"{ASTROLOC_CACHE}/results_baseline.json"]),
    ("Full fine-tune (40ep)", COLORS["finetuned"], [f"{ASTROLOC_CACHE}/results_finetuned.json"]),
    ("LoRA fine-tune (40ep) -- best", COLORS["lora"], [
        f"{ASTROLOC_CACHE}/results_lora_finetuned_a.json",
        f"{ASTROLOC_CACHE}/results_lora_finetuned_b.json",
    ]),
    ("Nano v1 (zoom-9-only)", COLORS["nano"], [f"{NANO_CACHE}/results_nano-dinov2s-dynamic-512d.json"]),
    ("Nano v2 (multi-zoom + LoRA)", "#ca8a04", [f"{NANO_CACHE}/results_nano-v2-multizoom-lora-dynamic.json"]),
]


def load_per_region(paths):
    combined = {}
    for p in paths:
        if not os.path.exists(p):
            return None
        combined.update(json.load(open(p)))
    return combined


REGION_ORDER = ["Alps", "Texas", "Toshka Lakes", "Amazon", "Napa", "Gobi"]

K_VALUES = ["1", "5", "10", "100"]


def bar_chart_recall(models, title, filename):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_models = len(models)
    x = range(len(K_VALUES))
    width = 0.8 / n_models
    for i, (label, recalls, color) in enumerate(models):
        offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
        vals = [recalls[k] for k in K_VALUES]
        bars = ax.bar(offsets, vals, width=width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"R@{k}" for k in K_VALUES])
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"wrote {filename}")


# Graph 1: training progression on the full-size model
bar_chart_recall(
    [
        ("Baseline (pretrained, no fine-tune)", baseline_r, COLORS["baseline"]),
        ("Full fine-tune (40ep)", finetuned_r, COLORS["finetuned"]),
        ("LoRA fine-tune (40ep) -- best", lora_r, COLORS["lora"]),
    ],
    "Retrieval recall, averaged across 6 held-out regions\n(DINOv2-base + SALAD, 106M params)",
    "01_training_progression.png",
)

# Graph 2: full-size best vs nano v1 vs nano v2 (v2 added once its eval exists)
_g2_models = [
    ("Full-size (LoRA, 106M params, 2048-dim)", lora_r, COLORS["lora"]),
    ("Nano v1 (zoom-9-only, no LoRA)", nano_r, COLORS["nano"]),
]
if nano_v2_r is not None:
    _g2_models.append(("Nano v2 (multi-zoom + LoRA)", nano_v2_r, "#ca8a04"))
bar_chart_recall(
    _g2_models,
    "Accuracy cost of the efficient (nano) variant",
    "02_fullsize_vs_nano.png",
)

# Graph 3: efficiency (params + descriptor size) bar chart
fig, ax1 = plt.subplots(figsize=(7.5, 5))
labels = ["Full-size (DINOv2-base)\n2048-dim descriptor", "Nano (DINOv2-tiny)\n512-dim descriptor"]
params = [106, 27.2]
x = range(len(labels))
bars = ax1.bar(x, params, color=[COLORS["lora"], COLORS["nano"]], width=0.5)
for b, v in zip(bars, params):
    ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}M", ha="center", fontsize=10, fontweight="bold")
ax1.set_ylabel("Model parameters (M)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 120)
ax1.set_title("Model size: full-size vs nano")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_efficiency.png"), dpi=150)
plt.close(fig)
print("wrote 03_efficiency.png")

# Graph 4: OD error summary (two-frame simulated OD velocity recovery)
# Numbers from real integration/two_frame_od_demo.py runs (one real localized
# fix + simulated frame 2 + snap-to-nearest-real-tile, per model/checkpoint).
fig, ax = plt.subplots(figsize=(7.5, 5))
od_labels = ["Full-size model\n(4.2km tile snap)", "Nano v1\n(58.7km tile snap)", "Nano v2\n(19.1km tile snap)"]
od_errors_ms = [13.3, 483.2, 116.4]
od_colors = [COLORS["lora"], COLORS["nano"], "#ca8a04"]
bars = ax.bar(od_labels, od_errors_ms, color=od_colors, width=0.5)
for b, v in zip(bars, od_errors_ms):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.1f} m/s", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Velocity recovery error (m/s)")
ax.set_title("Two-frame simulated OD: velocity error\n(true speed ~7,657 m/s in both cases)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_od_velocity_error.png"), dpi=150)
plt.close(fig)
print("wrote 04_od_velocity_error.png")

# Graph 5: per-region ground (coordinate) error, grouped by region, bars = models.
# Only includes models whose results file currently exists -- nano v2 slots in
# automatically once its eval has run.
available = [(label, color, load_per_region(paths)) for label, color, paths in PER_REGION_MODELS]
available = [(label, color, data) for label, color, data in available if data is not None]

fig, ax = plt.subplots(figsize=(12, 6))
n_models = len(available)
x = range(len(REGION_ORDER))
width = 0.8 / n_models
for i, (label, color, data) in enumerate(available):
    offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
    vals = [data[r]["coords"]["median_coord_error_km"] for r in REGION_ORDER]
    ax.bar(offsets, vals, width=width, label=label, color=color)
ax.set_xticks(list(x))
ax.set_xticklabels(REGION_ORDER)
ax.set_ylabel("Median coordinate error (km, retrieval-only, log scale)")
ax.set_yscale("log")
ax.set_title("Ground error by region and model")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_ground_error_by_region.png"), dpi=150)
plt.close(fig)
print("wrote 05_ground_error_by_region.png")

# Graph 6: nano CPU RAM consumption by region (nano v1 always; v2 added once ready)
nano_runs = [
    ("Nano v1 (zoom-9-only)", COLORS["nano"], f"{NANO_CACHE}/results_nano-dinov2s-dynamic-512d.json"),
    ("Nano v2 (multi-zoom + LoRA)", "#ca8a04", f"{NANO_CACHE}/results_nano-v2-multizoom-lora-dynamic.json"),
]
nano_available = [(label, color, json.load(open(p))) for label, color, p in nano_runs if os.path.exists(p)]

fig, ax = plt.subplots(figsize=(10, 6))
n_models = len(nano_available)
x = range(len(REGION_ORDER))
width = 0.8 / max(n_models, 1)
for i, (label, color, data) in enumerate(nano_available):
    offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
    vals = [data[r]["resource_use_during_eval"]["peak_rss_mb"] / 1024 for r in REGION_ORDER]
    bars = ax.bar(offsets, vals, width=width, label=label, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(REGION_ORDER)
ax.set_ylabel("Peak CPU RAM during eval (GB)")
ax.set_title("Nano model: CPU RAM consumption by region")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_nano_ram_by_region.png"), dpi=150)
plt.close(fig)
print("wrote 06_nano_ram_by_region.png")

print("\n=== Summary numbers used ===")
print("Baseline:", baseline_r, f"median_err={baseline_err:.1f}km")
print("Full fine-tune:", finetuned_r, f"median_err={finetuned_err:.1f}km")
print("LoRA (best):", lora_r, f"median_err={lora_err:.1f}km")
print("Nano v1:", nano_r, f"median_err={nano_err:.1f}km")
if nano_v2_r is not None:
    print("Nano v2:", nano_v2_r, f"median_err={nano_v2_err:.1f}km")
print(f"\nModels included in per-region graphs: {[l for l, _, _ in available]}")
print(f"Nano runs included in RAM graph: {[l for l, _, _ in nano_available]}")
