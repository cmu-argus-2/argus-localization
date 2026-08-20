"""Interactive single-image demo: pick a region, a query image, and a trained
retriever, see the top-k retrieved candidates, choose one to match, then see
it carried all the way through to an OD position solve.

No such single-image, full-chain demo existed before this: astroloc/eval/evaluate.py
measures retrieval only (top-1 tile centroid) across many queries at once,
scripts/evaluate_astroloc_matched.py measures the real matched pipeline but
also as an aggregate over many queries, and integration/od_integration_test.py's
OD solve only ever ran on a synthetic trajectory. This wires one real image
through retrieval -> user-chosen candidate -> SIFT-LightGlue match ->
georeferencing -> bearing/landmark conversion -> a real position solve, all
in one interactive pass.

Runs everything on CPU by default: both GPUs are occupied by the two 100-epoch
training runs this session kicked off (astroloc/training/train.py, dynamic +
static, ~/logs/train_dynamic_v2.log / train_lora_v2.log), and this demo has a
human waiting on each click rather than needing throughput.

OD caveat shown in the UI, not just here: a single image gives one epoch of
bearing measurements, which constrains position AND attitude jointly (nothing
links to a second frame, so velocity/orbit shape is not observable) --
see integration/pose_resection.py's docstring for why this is the honest
scope, versus the multi-frame, known-attitude OD that FSW-Payload's real
Ceres solver runs. Camera intrinsics are Argus's own CameraModel as a stand-in
(these EarthLoc/GAPE photos weren't shot with Argus's camera; same caveat
integration/od_integration_test.py already documents).

Run: streamlit run astroloc/demo/streamlit_app.py
"""

import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core.path_setup import ensure_repo_root_first

ensure_repo_root_first(_REPO_ROOT)
os.environ.setdefault("TORCH_HOME", "/mnt/sdc1/astroloc/reference_db/astroloc_train/cache/torch")

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from brahe.epoch import Epoch
from sensors.camera_model import CameraModelManager

from astroloc.models.dinov2_salad import DinoV2SaladModel, DinoV2SaladRetriever
from database.reference_database import ReferenceDatabase, dedup_search
from georeference.georeferencer import Georeferencer
from index.faiss_index import FaissFlatIndex
from integration.batchopt_adapter import to_batchopt_measurements
from integration.pose_resection import solve_single_frame_pose
from matchers.sift_lightglue_matcher import SiftLightGlueMatcher
from scripts.evaluate import REGIONS, find_positive_tile_ids, footprint_bbox, haversine_km, load_scoped_queries

TRAIN_DIR = "/mnt/sdc1/astroloc/reference_db/astroloc_train"
CACHE_ROOT = os.path.join(TRAIN_DIR, "eval_cache")
QUERIES_DIR = "/mnt/sdc1/astroloc/data/queries"
DEVICE = "cpu"  # deployment-relevant regime, also lets this run alongside GPU training jobs

NANO_TRAIN_DIR = "/mnt/sdc1/astroloc/reference_db/nano_train"
NANO_CACHE_ROOT = os.path.join(NANO_TRAIN_DIR, "eval_cache")

MODELS = {
    "Baseline (pretrained, no fine-tune)": {"checkpoint": None, "cache_key": "baseline"},
    "Full fine-tune (static, 40ep)": {"checkpoint": f"{TRAIN_DIR}/checkpoints/final.pt", "cache_key": "finetuned"},
    "LoRA (static, 40ep, lr=2e-4) -- prior best": {"checkpoint": f"{TRAIN_DIR}/checkpoints_lora/final.pt", "cache_key": "lora_finetuned"},
    "LoRA (dynamic batching, 40ep)": {"checkpoint": f"{TRAIN_DIR}/checkpoints_dynamic/final.pt", "cache_key": "dynamic_finetuned"},
    "LoRA (dynamic, 100ep, paper lr=5e-5) [new]": {"checkpoint": f"{TRAIN_DIR}/checkpoints_dynamic_v2/final.pt", "cache_key": "dynamic_finetuned_v2"},
    "LoRA (static, 100ep, paper lr=5e-5) [new]": {"checkpoint": f"{TRAIN_DIR}/checkpoints_lora_v2/final.pt", "cache_key": "lora_finetuned_v2"},
    "Nano v1 (DINOv2-tiny, zoom-9-only, 27.2M)": {
        "checkpoint": f"{NANO_TRAIN_DIR}/checkpoints/final.pt",
        "cache_key": "nano-dinov2s-dynamic-512d",
        "cache_root": NANO_CACHE_ROOT,
    },
    "Nano v2 (DINOv2-tiny, multi-zoom + LoRA, 27.2M)": {
        "checkpoint": f"{NANO_TRAIN_DIR}/checkpoints_v2/final.pt",
        "cache_key": "nano-v2-multizoom-lora-dynamic",
        "cache_root": NANO_CACHE_ROOT,
    },
}


def resolve_cache_dir(cache_key: str, region: str, cache_root: str = CACHE_ROOT) -> str:
    region_dir = region.replace(" ", "_")
    if cache_key == "lora_finetuned":
        sub = "lora_finetuned_a" if region in ("Alps", "Texas", "Toshka Lakes") else "lora_finetuned_b"
        return os.path.join(cache_root, sub, region_dir)
    return os.path.join(cache_root, cache_key, region_dir)


def available_models_for_region(region: str) -> dict:
    out = {}
    for name, cfg in MODELS.items():
        ckpt_ok = cfg["checkpoint"] is None or os.path.exists(cfg["checkpoint"])
        cache_dir = resolve_cache_dir(cfg["cache_key"], region, cfg.get("cache_root", CACHE_ROOT))
        if ckpt_ok and os.path.exists(os.path.join(cache_dir, "tiles.json")):
            out[name] = {**cfg, "cache_dir": cache_dir}
    return out


@st.cache_resource(show_spinner="Loading retriever + reference database...")
def load_db(checkpoint: str | None, cache_dir: str):
    if checkpoint is None:
        model = DinoV2SaladModel(pretrained=True)
        retriever = DinoV2SaladRetriever(model, device=DEVICE)
    else:
        retriever = DinoV2SaladRetriever.from_checkpoint(checkpoint, device=DEVICE)
    index = FaissFlatIndex(retriever.descriptor_dim)
    db = ReferenceDatabase.load(cache_dir, retriever, index)
    return db


@st.cache_resource(show_spinner=False)
def load_matcher_georef():
    matcher = SiftLightGlueMatcher(max_num_keypoints=1024, img_size=512, max_ransac_iters=3, min_inliers=30, device=DEVICE)
    return matcher, Georeferencer()


@st.cache_resource(show_spinner=False)
def load_camera_model():
    return CameraModelManager()["x+"]


@st.cache_data(show_spinner=False)
def list_queries(region: str):
    center_lat, center_lon = REGIONS[region]
    queries = load_scoped_queries(QUERIES_DIR, center_lat, center_lon, 2500)
    return queries


@st.cache_data(show_spinner=False)
def load_results_table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(CACHE_ROOT, "results_*.json"))):
        model_name = os.path.basename(path)[len("results_"):-len(".json")]
        if model_name == "smoketest":
            continue
        with open(path) as f:
            data = json.load(f)
        for region, r in data.items():
            rows.append(
                {
                    "model": model_name,
                    "region": region,
                    "R@1": round(r["recalls"]["1"], 1),
                    "R@5": round(r["recalls"]["5"], 1),
                    "R@10": round(r["recalls"]["10"], 1),
                    "R@100": round(r["recalls"]["100"], 1),
                    "median_coord_err_km (retrieval-only)": (
                        round(r["coords"]["median_coord_error_km"], 1)
                        if r["coords"]["median_coord_error_km"] is not None
                        else None
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["region", "model"]).reset_index(drop=True)


VERIFIED_CACHE_ROOT = os.path.join(CACHE_ROOT, "verified_queries")
SCAN_SAMPLE_CAP = 40
SCAN_TOP_K = 10


def _verified_cache_path(cache_key: str, region: str) -> str:
    return os.path.join(VERIFIED_CACHE_ROOT, cache_key, f"{region.replace(' ', '_')}.json")


def scan_verified_queries(region: str, model_cfg: dict, progress_cb=None) -> dict:
    """One-time (then cached to disk forever) scan: which queries in this region
    actually produce a >=min_inliers match with this model, and why the rest
    don't -- split into 'retrieval miss' (correct tile never made the top-k
    shortlist) vs 'matcher shortfall' (it was retrieved, but no candidate's
    SIFT-LightGlue match cleared min_inliers). This is the direct fix for
    picking a query blind and landing on 0 inliers: the query picker below
    only lists queries this scan already confirmed will produce a fix.
    """
    cache_path = _verified_cache_path(model_cfg["cache_key"], region)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    db = load_db(model_cfg["checkpoint"], model_cfg["cache_dir"])
    matcher, _ = load_matcher_georef()
    min_inliers = matcher.min_inliers

    import random

    sample = list_queries(region)[:]
    random.Random(0).shuffle(sample)
    sample = sample[:SCAN_SAMPLE_CAP]

    db_tiles = list(db.tiles.values())
    db_bboxes = np.array([footprint_bbox(t) for t in db_tiles])

    verified, n_retrieval_miss, n_matcher_shortfall = [], 0, 0
    for i, q in enumerate(sample):
        if progress_cb:
            progress_cb(i, len(sample), q.tile_id)
        positives = find_positive_tile_ids(q, db_tiles, db_bboxes, iou_threshold=0.2)
        if not positives:
            continue  # no ground-truth positive in this db at all (e.g. over the sea) -- unscoreable, not a failure
        frame = np.array(Image.open(q.image_path).convert("RGB"))
        descriptor = db.retriever.embed(frame)
        ranked = dedup_search(db.index, descriptor, SCAN_TOP_K)
        retrieved_ids = {tid for tid, _ in ranked}
        if not (retrieved_ids & set(positives)):
            n_retrieval_miss += 1
            continue

        best_inliers = 0
        for tid, _sim in ranked:
            tile = db.tiles[tid]
            tile_image = np.array(Image.open(tile.image_path).convert("RGB"))
            m = matcher.match(frame, tile_image, tile_id=tid)
            best_inliers = max(best_inliers, m.num_inliers)
            if best_inliers >= min_inliers:
                break
        if best_inliers >= min_inliers:
            verified.append({"tile_id": q.tile_id, "best_inliers": best_inliers})
        else:
            n_matcher_shortfall += 1

    result = {
        "verified": sorted(verified, key=lambda v: -v["best_inliers"]),
        "stats": {
            "n_sampled": len(sample),
            "n_retrieval_miss": n_retrieval_miss,
            "n_matcher_shortfall": n_matcher_shortfall,
            "n_verified": len(verified),
        },
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


st.set_page_config(page_title="Argus Localization -> OD Demo", layout="wide")
st.title("Argus Localization -> OD Demo")
st.caption(
    "Retrieval -> user-chosen candidate -> SIFT-LightGlue match -> georeferencing -> "
    "single-frame OD pose solve, on one real image at a time. Running on CPU (GPUs are "
    "busy training). See astroloc/demo/streamlit_app.py docstring for scope/caveats."
)

with st.expander("Model x region statistics (from saved eval runs)", expanded=True):
    results_df = load_results_table()
    if results_df.empty:
        st.info("No results_*.json found under eval_cache/ yet.")
    else:
        st.dataframe(results_df, use_container_width=True, hide_index=True)

st.sidebar.header("Setup")
region = st.sidebar.selectbox("Region", list(REGIONS.keys()))

models_here = available_models_for_region(region)
if not models_here:
    st.sidebar.error(f"No model has a built reference-db cache for {region} yet.")
    st.stop()
model_name = st.sidebar.selectbox("Model", list(models_here.keys()))
model_cfg = models_here[model_name]

scan_cache_path = _verified_cache_path(model_cfg["cache_key"], region)
if not os.path.exists(scan_cache_path):
    st.sidebar.warning(
        f"Query images for {region} / {model_name} haven't been verified yet -- "
        f"picking one blind risks a 0-inlier match."
    )
    if st.sidebar.button("Scan for verified-good queries (one-time, ~a few minutes)"):
        progress = st.sidebar.progress(0.0, text="Starting scan...")

        def _cb(i, n, tile_id):
            progress.progress((i + 1) / n, text=f"Scanning {i+1}/{n}: {tile_id}")

        scan_verified_queries(region, model_cfg, progress_cb=_cb)
        st.rerun()
    st.sidebar.info("Scan not run yet -- pick a different region/model, or click the button above.")
    st.stop()

scan = scan_verified_queries(region, model_cfg)
stats = scan["stats"]
all_queries_by_id = {q.tile_id: q for q in list_queries(region)}
query_options = {v["tile_id"]: all_queries_by_id[v["tile_id"]] for v in scan["verified"] if v["tile_id"] in all_queries_by_id}

st.sidebar.caption(
    f"Verified {stats['n_verified']}/{stats['n_sampled']} sampled queries produce a fix "
    f"({stats['n_retrieval_miss']} retrieval misses, {stats['n_matcher_shortfall']} matcher shortfalls) "
    f"-- only verified ones are listed below."
)
if not query_options:
    st.sidebar.error(
        f"None of the {stats['n_sampled']} sampled queries produced a fix with this model in "
        f"{region}. Try a different model (some are much weaker -- see the stats table above)."
    )
    st.stop()

if "query_choice" not in st.session_state or st.session_state.get("query_region") != region or st.session_state.query_choice not in query_options:
    st.session_state.query_choice = next(iter(query_options))
    st.session_state.query_region = region

col_pick, col_rand = st.sidebar.columns([3, 1])
query_labels = {tid: f"{tid}  (best={next(v['best_inliers'] for v in scan['verified'] if v['tile_id'] == tid)} inliers)" for tid in query_options}
query_id = col_pick.selectbox(
    "Query image (verified-good only)",
    list(query_options.keys()),
    format_func=lambda tid: query_labels[tid],
    key="query_choice",
)
if col_rand.button("Random"):
    import random

    st.session_state.query_choice = random.choice(list(query_options.keys()))
    st.rerun()

query = query_options[query_id]
top_k = st.sidebar.slider("Top-k candidates", 3, 15, 10)
run = st.sidebar.button("Run retrieval", type="primary")

if run:
    st.session_state.did_run = True
    st.session_state.run_region = region
    st.session_state.run_model = model_name
    st.session_state.run_query = query_id
    st.session_state.run_topk = top_k

if not st.session_state.get("did_run"):
    st.info("Pick a region, model, and query image, then click **Run retrieval** in the sidebar.")
    st.stop()

region, model_name, query_id, top_k = (
    st.session_state.run_region,
    st.session_state.run_model,
    st.session_state.run_query,
    st.session_state.run_topk,
)
model_cfg = available_models_for_region(region)[model_name]
query = {q.tile_id: q for q in list_queries(region)}[query_id]

db = load_db(model_cfg["checkpoint"], model_cfg["cache_dir"])
matcher, georef = load_matcher_georef()

query_frame = np.array(Image.open(query.image_path).convert("RGB"))

left, right = st.columns([1, 2])
with left:
    st.subheader("Query image")
    st.image(query_frame, use_container_width=True, caption=query.tile_id)
    gt_center = query.corners_latlon.mean(axis=0)
    st.caption(f"Ground-truth footprint centroid: {gt_center[0]:.3f}, {gt_center[1]:.3f}")

with right:
    st.subheader(f"Top-{top_k} retrieved candidates")
    st.caption(
        "All candidates are matched up front so the best one can be highlighted -- "
        "picking blind is how you end up with a 0-inlier candidate."
    )
    candidates = db.retrieve(query_frame, top_k)

    match_cache_key = (region, model_name, query_id, top_k)
    if st.session_state.get("match_cache_key") != match_cache_key:
        with st.spinner(f"Matching against all {len(candidates)} candidates to find the best one..."):
            match_results = []
            for tile, sim in candidates:
                tile_image = np.array(Image.open(tile.image_path).convert("RGB"))
                m = matcher.match(query_frame, tile_image, tile_id=tile.tile_id)
                match_results.append({"tile": tile, "sim": sim, "match": m, "tile_image": tile_image})
        st.session_state.match_cache_key = match_cache_key
        st.session_state.match_results = match_results
    match_results = st.session_state.match_results

    best_idx = max(range(len(match_results)), key=lambda i: match_results[i]["match"].num_inliers)
    best_inliers = match_results[best_idx]["match"].num_inliers

    cand_cols = st.columns(5)
    cand_labels = []
    for i, mr in enumerate(match_results):
        tile, sim, m = mr["tile"], mr["sim"], mr["match"]
        is_best = i == best_idx
        fix_ok = m.num_inliers >= 30
        tag = "BEST" if is_best else ("ok" if fix_ok else "would fail")
        label = f"[{tag}] {tile.tile_id}  (sim={sim:.3f}, inliers={m.num_inliers})"
        cand_labels.append(label)
        with cand_cols[i % 5]:
            caption = f"#{i+1} sim={sim:.3f}  inliers={m.num_inliers}"
            caption += "  ⭐ BEST" if is_best else ("" if fix_ok else "  ⚠️ below min_inliers")
            st.image(tile.image_path, use_container_width=True, caption=caption)

    if best_inliers < 30:
        st.warning(
            f"Even the best candidate only has {best_inliers} inliers (below min_inliers=30) -- "
            "this query likely won't produce a fix with this model/region. Try 'Random' for a "
            "different query, or a different model."
        )

    chosen_label = st.selectbox(
        "Choose a candidate (defaults to the best match found -- override to explore a worse one)",
        cand_labels,
        index=best_idx,
    )
    chosen_i = cand_labels.index(chosen_label)
    chosen = match_results[chosen_i]
    chosen_tile, match, tile_image = chosen["tile"], chosen["match"], chosen["tile_image"]

tie_points = georef.make_tie_points(match, chosen_tile, tile_image.shape)
query_footprint = georef.estimate_query_footprint(match, chosen_tile, query_frame.shape, tile_image.shape)

st.divider()
st.subheader("Match + coordinates")
c1, c2, c3 = st.columns(3)
c1.metric("Inliers", match.num_inliers)
c1.metric("Fix?", "yes" if match.num_inliers >= 30 else "no (below min_inliers=30)")

if tie_points:
    pred_center = np.array([[tp.lat, tp.lon] for tp in tie_points]).mean(axis=0)
    err_km = haversine_km(gt_center[0], gt_center[1], pred_center[0], pred_center[1])
    c2.metric("Predicted coords (tie-point centroid)", f"{pred_center[0]:.3f}, {pred_center[1]:.3f}")
    c3.metric("Localization error vs ground truth", f"{err_km:.2f} km")
else:
    st.warning("No inlier tie points -- can't estimate coordinates or run OD for this candidate.")
    st.stop()

st.divider()
st.subheader("OD: single-frame position resection")
st.caption(
    "Position + attitude solved jointly from this frame's bearing/landmark correspondences "
    "(velocity is not observable from one epoch -- see integration/pose_resection.py). "
    "Error shown is the fit's own RMS angular residual, not error against known spacecraft "
    "truth, which doesn't exist for these photos."
)

ts = query.timestamp
epoch = Epoch(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), 0, 0, 0.0)
cam = load_camera_model()


class _FakeResult:
    status = "fix"


result_obj = _FakeResult()
result_obj.tie_points = tie_points
measurements, _ = to_batchopt_measurements(result_obj, epoch, cam, query_frame.shape[:2])
bearing_body = measurements[:, 1:4]
landmark_eci_m = measurements[:, 4:7]

pose = solve_single_frame_pose(bearing_body, landmark_eci_m)

if not pose.get("success"):
    st.error(f"OD solve did not converge / not enough points: {pose}")
else:
    o1, o2, o3, o4 = st.columns(4)
    pos = pose["position_eci_km"]
    o1.metric("Solved ECI position (km)", f"[{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")
    o2.metric("Solved altitude", f"{pose['altitude_km']:.1f} km")
    o3.metric("RMS angular residual", f"{pose['rms_angular_residual_deg']:.2f} deg")
    o4.metric("# measurements used", pose["num_measurements"])
    st.caption(
        f"solver cost={pose['cost']:.3g}, nfev={pose['nfev']}. Typical ISS altitude is ~400-420km; "
        "large deviations reflect the stacked approximations here (non-Argus camera intrinsics, "
        "no real attitude telemetry), not necessarily a bad retrieval/match."
    )

st.divider()
st.subheader("OD: two-frame simulated orbit (velocity-observable)")
st.caption(
    "Frame 1 is real (the match above). Frame 2 is simulated: assume nadir-pointing + an input "
    "altitude/inclination at frame 1's real ground point, propagate forward by dt seconds with a "
    "real two-body integrator, then snap the predicted ground point to the nearest REAL reference "
    "tile -- no second photo or matching needed. The solver then finds the velocity connecting the "
    "two fixes. This is a round-trip self-consistency check (frame 2 comes from the same orbital "
    "model being solved for), not independent-measurement recovery -- see "
    "integration/orbit_simulator.py's docstring. Speed error below is against the simulator's own "
    "known input velocity, which IS a real ground truth here (unlike the single-frame section above)."
)

od2_c1, od2_c2 = st.columns(2)
altitude_km_input = od2_c1.slider("Assumed altitude (km)", 300.0, 500.0, 420.0, step=10.0)
dt_s_input = od2_c2.slider("Simulated dt to frame 2 (s)", 30.0, 600.0, 120.0, step=30.0)

if pose.get("success"):
    from integration.orbit_simulator import nearest_reference_tile, simulate_next_frame, solve_two_frame_od

    sim2 = simulate_next_frame(
        pred_center[0], pred_center[1], epoch, altitude_km=altitude_km_input, dt_s=dt_s_input
    )
    snapped_tile = nearest_reference_tile(list(db.tiles.values()), sim2["lat2_deg"], sim2["lon2_deg"])
    snapped_center = snapped_tile.corners_latlon.mean(axis=0)
    od2 = solve_two_frame_od(
        pred_center[0], pred_center[1], epoch,
        float(snapped_center[0]), float(snapped_center[1]), sim2["epoch2"],
        altitude_km=altitude_km_input,
    )
    true_speed_kms = float(np.linalg.norm(sim2["v1_true"]) / 1e3)

    if not od2["success"]:
        st.error(f"Two-frame OD solve did not converge: {od2}")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Solved speed", f"{od2['speed_solved_kms']:.4f} km/s")
        p2.metric("True (simulator) speed", f"{true_speed_kms:.4f} km/s")
        p3.metric("Speed error", f"{abs(od2['speed_solved_kms'] - true_speed_kms) * 1e3:.1f} m/s")
        p4.metric("Snapped tile", snapped_tile.tile_id)
        snap_km = haversine_km(sim2["lat2_deg"], sim2["lon2_deg"], snapped_center[0], snapped_center[1])
        st.caption(
            f"Frame 2 simulated ground point: {sim2['lat2_deg']:.3f}, {sim2['lon2_deg']:.3f} at "
            f"{sim2['epoch2']}. Snapped to a real tile {snap_km:.1f} km away. Direction residual: "
            f"{od2['direction_residual_rad']:.2e} rad."
        )
else:
    st.info("Two-frame OD needs a successful single-frame solve above first.")
