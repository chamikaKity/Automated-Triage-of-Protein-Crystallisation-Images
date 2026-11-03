"""Score the trivial benchmark (always predict the majority class) and log it.

The floor every real model must beat for the Q3 comparison table. Majority class is
determined from the train split only; scored on val only - test stays untouched, same as
every other model-selection step.
"""

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marco.data import CLASS_ORDER, load_split
from marco.metrics import recall_at_precision

SEED = 42


def main() -> None:
    splits_path = Path("data/splits.csv")
    log_path = Path("experiments/log.csv")

    train_df = load_split(splits_path, "train")
    val_df = load_split(splits_path, "val")
    majority_class = train_df["label"].value_counts().idxmax()

    t0 = time.time()
    preds = np.full(len(val_df), majority_class)
    accuracy = float((preds == val_df["label"].to_numpy()).mean())

    crystals_idx = CLASS_ORDER.index("Crystals")
    y_bin = (val_df["label"].to_numpy() == "Crystals").astype(int)
    # The trivial benchmark never predicts Crystals, so every image gets the same
    # (lowest possible) Crystals score - it can never clear any positive precision bar.
    crystals_scores = np.zeros(len(val_df))
    train_time_s = time.time() - t0

    t0 = time.time()
    metrics = {
        "crystals_pr_auc": float(average_precision_score(y_bin, crystals_scores)),
        "recall_at_p_min": recall_at_precision(y_bin, crystals_scores),
        "accuracy": accuracy,
    }
    infer_latency_ms = (time.time() - t0) / len(val_df) * 1000

    print(f"majority class (from train): {majority_class}")
    print(f"val: {metrics}")

    log_row = {
        "id": "trivial_benchmark_000",
        "date": date.today().isoformat(),
        "model": "Trivial benchmark (always predict majority class)",
        "what_changed": f"majority class = {majority_class}, no training",
        "hyperparameters": json.dumps({}),
        "seed": SEED,
        "validation_metrics": json.dumps({
            **metrics,
            "train_time_s": round(train_time_s, 4),
            "inference_latency_ms_per_image": round(infer_latency_ms, 4),
        }),
        "notes": "",
    }
    pd.DataFrame([log_row]).to_csv(log_path, mode="a", header=False, index=False)
    print(f"logged to {log_path}")


if __name__ == "__main__":
    main()
