"""Train the classical baseline (LightGBM on HOG+LBP features) and log every run.

Loads the fixed train/val split (data/splits.csv) - never touches test. Extracts and caches
HOG+LBP features once (src/marco/features.py), then runs a random hyperparameter search for
LightGBM, picking the best config by validation PR-AUC for Crystals - recall@fixed-precision
isn't usable for model selection yet since P_MIN hasn't been set (this run's PR curve is what
sets it, see docs/). Every config tried is appended to experiments/log.csv per rule 5.
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from marco.features import extract_features

SEED = 42
CLASS_ORDER = ["Clear", "Crystals", "Other", "Precipitate"]
N_CONFIGS = 24


def load_or_extract_features(df: pd.DataFrame, data_dir: Path, cache_path: Path, force: bool) -> np.ndarray:
    """Extract HOG+LBP features for every row in df, or load them from cache_path."""
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if np.array_equal(cached["image_id"], df["image_id"].to_numpy()):
            return cached["X"]
    X = np.stack([extract_features(data_dir / p) for p in df["path"]])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, image_id=df["image_id"].to_numpy())
    return X


def sample_hyperparams(rng: np.random.Generator) -> dict:
    """One random draw from the LightGBM search space."""
    return {
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "max_depth": int(rng.choice([-1, 4, 6, 8])),
        "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.1])),
        "n_estimators": int(rng.choice([100, 200, 400])),
        "min_child_samples": int(rng.choice([10, 20, 50])),
        "feature_fraction": float(rng.choice([0.6, 0.8, 1.0])),
        "bagging_fraction": float(rng.choice([0.6, 0.8, 1.0])),
        "bagging_freq": int(rng.choice([0, 1, 5])),
    }


def evaluate(model: LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Crystals PR-AUC (model-selection criterion) and overall accuracy (context only)."""
    proba = model.predict_proba(X_val)
    crystals_idx = list(model.classes_).index("Crystals")
    crystals_pr_auc = average_precision_score((y_val == "Crystals").astype(int), proba[:, crystals_idx])
    accuracy = (model.predict(X_val) == y_val).mean()
    return {"crystals_pr_auc": crystals_pr_auc, "accuracy": accuracy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features_baseline"))
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("experiments/models/baseline_lightgbm.joblib"))
    parser.add_argument("--n-configs", type=int, default=N_CONFIGS)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    splits = pd.read_csv(args.splits_path)
    train_df = splits[splits["split"] == "train"].reset_index(drop=True)
    val_df = splits[splits["split"] == "val"].reset_index(drop=True)

    print(f"extracting/loading features: {len(train_df)} train, {len(val_df)} val")
    t0 = time.time()
    X_train = load_or_extract_features(train_df, args.data_dir, args.cache_dir / "train_features.npz", args.force_features)
    X_val = load_or_extract_features(val_df, args.data_dir, args.cache_dir / "val_features.npz", args.force_features)
    print(f"features ready in {time.time() - t0:.1f}s (dim={X_train.shape[1]})")

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    rng = np.random.default_rng(args.seed)
    log_rows = []
    best = None

    for i in range(args.n_configs):
        params = sample_hyperparams(rng)
        model = LGBMClassifier(
            objective="multiclass", class_weight="balanced", random_state=args.seed,
            verbosity=-1, **params,
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time_s = time.time() - t0

        t0 = time.time()
        metrics = evaluate(model, X_val, y_val)
        infer_latency_ms = (time.time() - t0) / len(X_val) * 1000

        print(f"[{i + 1}/{args.n_configs}] {params} -> "
              f"crystals_pr_auc={metrics['crystals_pr_auc']:.3f} accuracy={metrics['accuracy']:.3f}")

        log_rows.append({
            "id": f"baseline_lgbm_{i:03d}",
            "date": date.today().isoformat(),
            "model": "LightGBM (HOG+LBP)",
            "what_changed": f"hyperparameter search config {i + 1}/{args.n_configs}",
            "hyperparameters": json.dumps(params),
            "seed": args.seed,
            "validation_metrics": json.dumps({
                **metrics,
                "train_time_s": round(train_time_s, 2),
                "inference_latency_ms_per_image": round(infer_latency_ms, 4),
            }),
            "notes": "",
        })

        if best is None or metrics["crystals_pr_auc"] > best["metrics"]["crystals_pr_auc"]:
            best = {"model": model, "params": params, "metrics": metrics,
                    "train_time_s": train_time_s, "infer_latency_ms": infer_latency_ms}

    pd.DataFrame(log_rows).to_csv(args.log_path, mode="a", header=False, index=False)
    print(f"\nlogged {len(log_rows)} configs to {args.log_path}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], args.model_out)
    print(f"best config saved to {args.model_out}: {best['params']}")
    print(f"best val crystals_pr_auc={best['metrics']['crystals_pr_auc']:.3f} "
          f"accuracy={best['metrics']['accuracy']:.3f} "
          f"train_time={best['train_time_s']:.1f}s "
          f"inference_latency={best['infer_latency_ms']:.3f}ms/image")

    preds = best["model"].predict(X_val)
    print("\nconfusion matrix (val, rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_val, preds, labels=CLASS_ORDER),
                        index=CLASS_ORDER, columns=CLASS_ORDER))

    proba = best["model"].predict_proba(X_val)
    crystals_idx = list(best["model"].classes_).index("Crystals")
    precision, recall, thresholds = precision_recall_curve(
        (y_val == "Crystals").astype(int), proba[:, crystals_idx]
    )
    print("\nval PR curve for Crystals (precision, recall, threshold) - use this to set P_MIN:")
    for p, r, t in list(zip(precision, recall, thresholds))[::max(1, len(thresholds) // 20)]:
        print(f"  precision={p:.3f} recall={r:.3f} threshold={t:.3f}")


if __name__ == "__main__":
    main()
