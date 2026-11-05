"""Feature-ablation experiment: does the classical baseline actually need both HOG and LBP?

Hypothesis (pre-stated, before running): the report's feature-importance chart shows the top 20
features split evenly 10/10 between HOG and LBP, so removing either signal type should hurt the
model - this is a controlled test of that claim, not just a hyperparameter re-search. Uses the
already-cached train/val features (data/features_baseline/), the same best hyperparameters as
the full-feature baseline (so only the feature set changes, nothing else), and never touches
test. Logs two rows to experiments/log.csv.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

SEED = 42
P_MIN = 0.30
CLASS_ORDER = ["Clear", "Crystals", "Other", "Precipitate"]
HOG_DIM = 1764  # see src/marco/features.py - 49 blocks x 2x2 cells x 9 orientations

# Same config as the saved full-feature baseline (experiments/models/baseline_lightgbm.joblib),
# so this is a clean ablation: only the feature set changes.
BEST_PARAMS = {
    "num_leaves": 63, "max_depth": -1, "learning_rate": 0.1, "n_estimators": 200,
    "min_child_samples": 50, "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
}


def recall_at_precision(y_true_binary: np.ndarray, scores: np.ndarray, p_min: float = P_MIN) -> float:
    precision, recall, _ = precision_recall_curve(y_true_binary, scores)
    achievable = recall[precision >= p_min]
    return float(achievable.max()) if len(achievable) else 0.0


def run_one(name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    model = LGBMClassifier(
        objective="multiclass", class_weight="balanced", random_state=SEED, verbosity=-1, **BEST_PARAMS,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)
    crystals_idx = list(model.classes_).index("Crystals")
    y_bin = (y_val == "Crystals").astype(int)
    crystals_scores = proba[:, crystals_idx]

    metrics = {
        "crystals_pr_auc": float(average_precision_score(y_bin, crystals_scores)),
        "recall_at_p_min": recall_at_precision(y_bin, crystals_scores),
        "accuracy": float((model.predict(X_val) == y_val).mean()),
        "n_features": X_train.shape[1],
    }
    cm = confusion_matrix(y_val, model.predict(X_val), labels=CLASS_ORDER).tolist()
    print(f"{name}: {metrics}")
    print(f"  confusion matrix (val, rows=true, cols=pred, order={CLASS_ORDER}): {cm}")
    return metrics, cm


def main() -> None:
    splits = pd.read_csv("data/splits.csv")
    train_df = splits[splits["split"] == "train"].reset_index(drop=True)
    val_df = splits[splits["split"] == "val"].reset_index(drop=True)

    train_cache = np.load("data/features_baseline/train_features.npz")
    val_cache = np.load("data/features_baseline/val_features.npz")
    assert np.array_equal(train_cache["image_id"], train_df["image_id"].to_numpy())
    assert np.array_equal(val_cache["image_id"], val_df["image_id"].to_numpy())
    X_train_full, X_val_full = train_cache["X"], val_cache["X"]
    y_train, y_val = train_df["label"].to_numpy(), val_df["label"].to_numpy()

    print(f"full feature dim: {X_train_full.shape[1]} (HOG={HOG_DIM}, LBP={X_train_full.shape[1] - HOG_DIM})")

    results = {}
    results["hog_only"] = run_one(
        "HOG-only", X_train_full[:, :HOG_DIM], y_train, X_val_full[:, :HOG_DIM], y_val,
    )
    results["lbp_only"] = run_one(
        "LBP-only", X_train_full[:, HOG_DIM:], y_train, X_val_full[:, HOG_DIM:], y_val,
    )

    log_rows = []
    for key, label in [("hog_only", "HOG-only (ablation)"), ("lbp_only", "LBP-only (ablation)")]:
        metrics, cm = results[key]
        log_rows.append({
            "id": f"baseline_ablation_{key}_000",
            "date": date.today().isoformat(),
            "model": f"LightGBM ({label})",
            "what_changed": f"feature ablation: {label} only, same hyperparameters as full-feature baseline",
            "hyperparameters": json.dumps(BEST_PARAMS),
            "seed": SEED,
            "validation_metrics": json.dumps({**metrics, "confusion_matrix": cm}),
            "notes": "Tests whether both HOG and LBP are needed (pre-stated hypothesis: yes, "
                     "per the full-model's top-20 feature importance split evenly between them).",
        })
    pd.DataFrame(log_rows).to_csv("experiments/log.csv", mode="a", header=False, index=False)
    print(f"\nlogged {len(log_rows)} rows to experiments/log.csv")

    print("\n--- comparison vs full-feature baseline (val, from experiments/log.csv / docs/ann_initial_training.md) ---")
    print("full HOG+LBP : crystals_pr_auc=0.652  recall_at_p_min=0.850")
    print(f"HOG-only     : crystals_pr_auc={results['hog_only'][0]['crystals_pr_auc']:.3f}  "
          f"recall_at_p_min={results['hog_only'][0]['recall_at_p_min']:.3f}")
    print(f"LBP-only     : crystals_pr_auc={results['lbp_only'][0]['crystals_pr_auc']:.3f}  "
          f"recall_at_p_min={results['lbp_only'][0]['recall_at_p_min']:.3f}")


if __name__ == "__main__":
    main()
