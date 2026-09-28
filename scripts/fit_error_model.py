"""Fits a per-model least-squares noise model, log(error_km) ~ b0 + b1*num_inliers
+ b2*retrieval_similarity, from scripts/collect_error_dataset.py's output
(output/error_dataset/combined.jsonl). Only "fix" rows carry a real
georeferenced error_km -- "no_fix"/"error" rows have no coordinate estimate,
so they're excluded from the regression itself (fix rate is reported
separately, since it's a different kind of reliability signal than the error
model).

Log-space fit because error_km is strictly positive and right-skewed (a few
badly-mismatched fixes can be 100s of km off while most cluster tightly) --
a log-normal noise model is the standard choice here, and gives OD a
straightforward predicted-error-at-a-given-confidence formula:
    error_km_typical(num_inliers, similarity) = exp(b0 + b1*num_inliers + b2*similarity)
with residual_std (in log space) describing the spread around that typical
value for a covariance/noise weighting, not just a single fixed constant.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def fit_one(rows: list[dict]) -> dict:
    fixes = [r for r in rows if r["status"] == "fix" and r["error_km"] and r["error_km"] > 0]
    y = np.log(np.array([r["error_km"] for r in fixes]))
    X = np.column_stack([
        np.ones(len(fixes)),
        np.array([r["num_inliers"] for r in fixes]),
        np.array([r["retrieval_similarity"] for r in fixes]),
    ])
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    resid = y - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "n_fix": len(fixes),
        "n_total": len(rows),
        "fix_rate_pct": 100.0 * len(fixes) / len(rows) if rows else 0.0,
        "b0_intercept": float(coef[0]),
        "b1_num_inliers": float(coef[1]),
        "b2_retrieval_similarity": float(coef[2]),
        "r_squared": float(r2),
        "log_residual_std": float(resid.std(ddof=3)),
        "median_error_km": float(np.median([r["error_km"] for r in fixes])),
        "mean_error_km": float(np.mean([r["error_km"] for r in fixes])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="output/error_dataset/combined.jsonl")
    ap.add_argument("--out", default="output/error_dataset/error_model.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dataset)]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    results = {}
    for model, model_rows in sorted(by_model.items()):
        fit = fit_one(model_rows)
        results[model] = fit
        print(f"\n=== {model} (n={fit['n_total']}, {fit['n_fix']} fixes, fix_rate={fit['fix_rate_pct']:.1f}%) ===")
        print(f"  log(error_km) = {fit['b0_intercept']:.4f} + {fit['b1_num_inliers']:.5f}*num_inliers "
              f"+ {fit['b2_retrieval_similarity']:.4f}*retrieval_similarity")
        print(f"  R^2={fit['r_squared']:.3f}  log_residual_std={fit['log_residual_std']:.3f}")
        print(f"  median_error_km={fit['median_error_km']:.2f}  mean_error_km={fit['mean_error_km']:.2f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved fitted models to {args.out}")


if __name__ == "__main__":
    main()
