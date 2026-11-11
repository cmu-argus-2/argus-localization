"""Draw the figures for PIPELINE-WALKTHROUGH.md from the output of pipeline_trace.py.

    python visualize_trace.py --trace pipeline_walkthrough/data --out pipeline_walkthrough
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
GREEN, RED, BLUE, AMBER = "#1a9850", "#d73027", "#2c7fb8", "#fdae61"
DPI = 110


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def apply_h(h, pts):
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:]


def ordered(poly):
    """Order polygon corners by angle around their centre so they draw as a closed shape."""
    c = poly.mean(axis=0)
    return poly[np.argsort(np.arctan2(poly[:, 0] - c[0], poly[:, 1] - c[1]))]


def side_by_side(photo, tile, gap=40):
    h = max(photo.shape[0], tile.shape[0])
    canvas = np.full((h, photo.shape[1] + gap + tile.shape[1], 3), 255, np.uint8)
    canvas[: photo.shape[0], : photo.shape[1]] = photo
    oy = (h - tile.shape[0]) // 2
    canvas[oy: oy + tile.shape[0], photo.shape[1] + gap:] = tile
    return canvas, np.array([photo.shape[1] + gap, oy])


def draw(label, d, out):
    s = json.load(open(os.path.join(d, "summary.json")))
    t = np.load(os.path.join(d, "trace.npz"))
    photo = np.array(Image.open(os.path.join(d, "photo.jpg")).convert("RGB"))
    best = s["best_rank"]
    tile = np.array(Image.open(os.path.join(d, "tiles", f"{best:02d}.jpg")).convert("RGB"))
    p = f"{label}_"

    # 1. photo
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(photo); ax.set_title(f"Input photo, {photo.shape[1]} × {photo.shape[0]} px"); ax.axis("off")
    save(fig, out, p + "01_photo.png")

    # 2. preprocessed
    x = np.clip(t["preprocessed"] * STD + MEAN, 0, 1).transpose(1, 2, 0)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    axs[0].imshow(photo); axs[0].set_title(f"original {photo.shape[1]} px")
    axs[1].imshow(x, interpolation="nearest"); axs[1].set_title("what the retriever sees: 224 × 224")
    for a in axs: a.axis("off")
    save(fig, out, p + "02_preprocessed.png")

    # 3. fingerprint
    v = t["fingerprint"]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(np.arange(len(v)), v, width=1.0, color=BLUE)
    ax.set_xlim(0, len(v)); ax.set_xlabel("position in the fingerprint (0 – 2047)"); ax.set_ylabel("value")
    ax.set_title(f"Fingerprint: {len(v)} numbers, scaled to length 1")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, p + "03_fingerprint.png")

    # 4a. score distribution
    sc, c = t["all_scores"], s["candidates"]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(sc, bins=80, color="#bbbbbb")
    for cc in c:
        ax.axvline(cc["score"], color=GREEN if cc["correct"] else RED, lw=1)
    ax.set_xlabel("similarity to the photo's fingerprint (best of 4 rotations)"); ax.set_ylabel("tiles")
    ax.set_title(f"All {len(sc):,} tiles scored; top 15 marked (green = correct tile, red = wrong)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, p + "04a_scores.png")

    # 4b. top-15 grid
    fig, axs = plt.subplots(3, 5, figsize=(13, 8.6))
    for cc, a in zip(c, axs.flat):
        a.imshow(Image.open(os.path.join(d, "tiles", f"{cc['rank']:02d}.jpg")))
        col = GREEN if cc["correct"] else RED
        a.add_patch(Rectangle((0, 0), 1, 1, transform=a.transAxes, fill=False, ec=col, lw=5))
        a.set_title(f"#{cc['rank'] + 1}  score {cc['score']:.3f}\n{'correct' if cc['correct'] else 'wrong'}, "
                    f"{cc['km_from_truth']:,.0f} km away", fontsize=9, color=col)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Top 15 tiles returned by the search", fontsize=13)
    save(fig, out, p + "04b_top15.png")

    # 5. SIFT points
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    for a, img, kp, scl, name in ((axs[0], photo, t["photo_kp"], t["photo_scales"], "photo"),
                                  (axs[1], tile, t["tile_kp"], t["tile_scales"], f"tile #{best + 1}")):
        a.imshow(img)
        a.scatter(kp[:, 0], kp[:, 1], s=np.clip(scl, 1, 30) * 1.5, facecolors="none", edgecolors=AMBER, lw=0.7)
        a.set_title(f"SIFT points on the {name}: {len(kp)}"); a.axis("off")
    save(fig, out, p + "05_sift.png")

    # 6 + 7. LightGlue matches, then MAGSAC inliers
    canvas, off = side_by_side(photo, tile)
    q, tp, inl = t["query_pts"], t["tile_pts"], t["inlier_mask"].astype(bool)
    for name, title, colours in (
        ("06_lightglue.png", f"LightGlue pairs the points: {len(q)} matches between photo and tile #{best + 1}",
         [AMBER] * len(q)),
        ("07_inliers.png", f"Geometry check (MAGSAC): {inl.sum()} kept (green), {(~inl).sum()} rejected (red)",
         [GREEN if k else RED for k in inl]),
    ):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.imshow(canvas)
        order = np.argsort(inl) if name.startswith("07") else np.arange(len(q))
        for i in order:
            ax.plot([q[i, 0], tp[i, 0] + off[0]], [q[i, 1], tp[i, 1] + off[1]], color=colours[i], lw=0.6, alpha=0.8)
        ax.set_title(title); ax.axis("off")
        save(fig, out, p + name)

    # 7b. inliers per candidate
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar([f"#{cc['rank'] + 1}" for cc in c], [cc["inliers"] for cc in c],
           color=[GREEN if cc["correct"] else RED for cc in c])
    ax.axhline(s["min_inliers"], color="black", ls="--", lw=1)
    ax.text(len(c) - 0.5, s["min_inliers"] + 3, f"fix needs ≥ {s['min_inliers']}", ha="right", fontsize=9)
    ax.set_ylabel("inliers"); ax.set_xlabel("candidate tile (search rank)")
    ax.set_title(f"Inliers for each of the 15 candidates; the most wins (#{best + 1}, {c[best]['inliers']})")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, p + "07b_inliers_per_tile.png")

    # 8. photo footprint on the tile
    h = t["homography"]
    pw, ph = photo.shape[1], photo.shape[0]
    corners = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], float)
    fp_px = apply_h(h, corners)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(tile)
    ax.add_patch(Polygon(fp_px, closed=True, fill=False, ec=BLUE, lw=2.5))
    ax.scatter(*apply_h(h, np.array([[pw / 2, ph / 2]]))[0], c=BLUE, s=60, marker="x")
    ax.scatter(tp[inl, 0], tp[inl, 1], s=6, c=GREEN)
    xs, ys = np.r_[fp_px[:, 0], 0, tile.shape[1]], np.r_[fp_px[:, 1], 0, tile.shape[0]]
    ax.set_xlim(xs.min() - 20, xs.max() + 20); ax.set_ylim(ys.max() + 20, ys.min() - 20)
    ax.set_title("The fitted homography maps the photo's outline\n(blue) onto the tile; × = photo centre")
    ax.axis("off")
    save(fig, out, p + "08_footprint_on_tile.png")

    # 9. on the map
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for poly, col, name, ls in ((t["tile_corners"], "#777777", f"matched tile #{best + 1}", "-"),
                                (t["truth_corners"], GREEN, "true photo footprint", "-"),
                                (t["footprint"], BLUE, "estimated footprint", "--")):
        o = ordered(np.asarray(poly))
        ax.add_patch(Polygon(o[:, ::-1], closed=True, fill=False, ec=col, lw=2, ls=ls, label=name))
    tc, ec = s["truth_centre"], s["estimated_centre"]
    ax.scatter(tc[1], tc[0], c=GREEN, s=70, marker="o", zorder=3)
    ax.scatter(ec[1], ec[0], c=BLUE, s=70, marker="x", zorder=3)
    ax.plot([tc[1], ec[1]], [tc[0], ec[0]], color="black", lw=1)
    allp = np.vstack([t["tile_corners"], t["truth_corners"], t["footprint"]])
    ax.set_xlim(allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.5); ax.set_ylim(allp[:, 0].min() - 0.5, allp[:, 0].max() + 0.5)
    ax.set_aspect(1 / math.cos(math.radians(tc[0])))
    ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
    ax.set_title(f"On the map: centre error {s['error_km']:.1f} km")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    save(fig, out, p + "09_on_map.png")

    # 10. tie points
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(photo)
    ax.scatter(q[inl, 0], q[inl, 1], s=14, c=GREEN, edgecolors="white", lw=0.4)
    for k, (u, vv) in enumerate(q[inl][:5]):
        ax.annotate(str(k + 1), (u, vv), color="white", fontsize=10, fontweight="bold", xytext=(5, -5), textcoords="offset points")
    ax.set_title(f"{inl.sum()} tie points: each is a photo pixel\nwith a known latitude/longitude"); ax.axis("off")
    save(fig, out, p + "10_tie_points.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for label in sorted(os.listdir(args.trace)):
        d = os.path.join(args.trace, label)
        if os.path.isfile(os.path.join(d, "summary.json")):
            draw(label, d, args.out)
            print("drew", label)


if __name__ == "__main__":
    main()
