"""SVR (support vector regression) alternative to fit_error_model.py's linear
least-squares fit of log(error_km) ~ num_inliers + retrieval_similarity.

Reports 5-fold CROSS-VALIDATED R^2 as the headline number, not training R^2:
the linear fit already showed this target has very little real signal from
these two covariates (R^2 0.008-0.032), and an RBF-kernel SVR can easily look
good on training data by fitting individual noisy points rather than a real
pattern -- cross-validation is the honest check for whether that happened
here too, or whether SVR's nonlinearity actually recovers a real
relationship that linear regression missed.
"""

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def fit_one(rows: list[dict]) -> tuple[dict, SVR, StandardScaler]:
    fixes = [r for r in rows if r["status"] == "fix" and r["error_km"] and r["error_km"] > 0]
    X = np.column_stack([
        np.array([r["num_inliers"] for r in fixes]),
        np.array([r["retrieval_similarity"] for r in fixes]),
    ])
    y = np.log(np.array([r["error_km"] for r in fixes]))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Small grid over the hyperparameters that matter most for RBF-SVR;
    # scored by CV R^2 so the search itself can't overfit by picking params
    # that only look good on the training set.
    grid = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid={"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.1], "epsilon": [0.05, 0.1, 0.2]},
        cv=5, scoring="r2",
    )
    grid.fit(X_scaled, y)
    best = grid.best_estimator_

    cv_r2 = grid.best_score_
    best.fit(X_scaled, y)
    y_pred_train = best.predict(X_scaled)
    train_r2 = 1 - np.sum((y - y_pred_train) ** 2) / np.sum((y - y.mean()) ** 2)

    return {
        "n_fix": len(fixes),
        "n_total": len(rows),
        "fix_rate_pct": 100.0 * len(fixes) / len(rows) if rows else 0.0,
        "best_params": grid.best_params_,
        "cv_r_squared": float(cv_r2),
        "train_r_squared": float(train_r2),
        "median_error_km": float(np.median([r["error_km"] for r in fixes])),
    }, best, scaler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="output/error_dataset/combined.jsonl")
    ap.add_argument("--out-summary", default="output/error_dataset/error_model_svr.json")
    ap.add_argument("--out-model-dir", default="output/error_dataset")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dataset)]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    results = {}
    for model, model_rows in sorted(by_model.items()):
        fit, svr, scaler = fit_one(model_rows)
        results[model] = fit
        print(f"\n=== {model} (n={fit['n_total']}, {fit['n_fix']} fixes) ===")
        print(f"  best params: {fit['best_params']}")
        print(f"  5-fold CV R^2={fit['cv_r_squared']:.3f}  (train R^2={fit['train_r_squared']:.3f})")
        print(f"  median_error_km={fit['median_error_km']:.2f}")

        with open(f"{args.out_model_dir}/svr_model_{model}.pkl", "wb") as f:
            pickle.dump({"svr": svr, "scaler": scaler}, f)

    with open(args.out_summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {args.out_summary}, models to {args.out_model_dir}/svr_model_<model>.pkl")


if __name__ == "__main__":
    main()
