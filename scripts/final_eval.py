"""Evaluate one model family on the TEST split and log it - the ONLY script that reads test.

Run once per model (--model trivial/baseline/ann), not all together in one process: loading
lightgbm and torch in the same process is unreliable on this machine (segfault or hang,
depending on op order - see docs/ann_initial_training.md's note), so each invocation only
touches the one library it needs.

Refuses to run unless docs/expectation.md already exists (hard rule 8 - the pre-registered
expectation must be written before test results exist).
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

from marco.metrics import recall_at_precision

SEED = 42
# Duplicated from marco.data rather than imported: marco.data imports torch at module level,
# and importing torch anywhere in this process before joblib.load (the --model baseline path)
# crashes - see docs/ann_initial_training.md's note. Keeping this file free of any top-level
# torch dependency is what makes running baseline and ann as separate processes actually work.
CLASS_ORDER = ["Clear", "Crystals", "Other", "Precipitate"]


def load_split(splits_path: Path, split: str) -> pd.DataFrame:
    splits = pd.read_csv(splits_path)
    return splits[splits["split"] == split].reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, preds: np.ndarray, y_bin: np.ndarray, crystals_scores: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, preds, labels=CLASS_ORDER)
    return {
        "accuracy": float((preds == y_true).mean()),
        "crystals_pr_auc": float(average_precision_score(y_bin, crystals_scores)),
        "recall_at_p_min": recall_at_precision(y_bin, crystals_scores),
        "confusion_matrix": cm.tolist(),
    }


def eval_trivial(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict, str]:
    majority_class = train_df["label"].value_counts().idxmax()
    y_true = test_df["label"].to_numpy()
    preds = np.full(len(test_df), majority_class)
    y_bin = (y_true == "Crystals").astype(int)
    crystals_scores = np.zeros(len(test_df))
    return compute_metrics(y_true, preds, y_bin, crystals_scores), f"majority class = {majority_class}"


def eval_baseline(data_dir: Path, cache_dir: Path, model_path: Path, test_df: pd.DataFrame) -> tuple[dict, str]:
    import joblib

    from marco.features import extract_features

    cache_path = cache_dir / "test_features.npz"
    X_test = None
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["image_id"], test_df["image_id"].to_numpy()):
            X_test = cached["X"]
    if X_test is None:
        X_test = np.stack([extract_features(data_dir / p) for p in test_df["path"]])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X_test, image_id=test_df["image_id"].to_numpy())

    model = joblib.load(model_path)
    y_true = test_df["label"].to_numpy()
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    crystals_idx = list(model.classes_).index("Crystals")
    y_bin = (y_true == "Crystals").astype(int)
    return compute_metrics(y_true, preds, y_bin, proba[:, crystals_idx]), f"model={model_path.name}"


def eval_ann(data_dir: Path, model_path: Path, history_path: Path, test_df: pd.DataFrame) -> tuple[dict, str]:
    import torch
    from torch.utils.data import DataLoader

    from marco.data import MarcoDataset
    from marco.model import build_model

    params = json.load(open(history_path))["params"]
    model = build_model(params["backbone"], fine_tune_degree=params.get("fine_tune_degree", "frozen"))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loader = DataLoader(MarcoDataset(test_df, data_dir, train=False), batch_size=32, shuffle=False, num_workers=0)
    all_probs, all_label_idx = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_probs.append(torch.softmax(model(imgs), dim=1).numpy())
            all_label_idx.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_label_idx = np.concatenate(all_label_idx)

    y_true = np.array(CLASS_ORDER)[all_label_idx]
    preds = np.array(CLASS_ORDER)[all_probs.argmax(axis=1)]
    crystals_idx = CLASS_ORDER.index("Crystals")
    y_bin = (all_label_idx == crystals_idx).astype(int)
    return compute_metrics(y_true, preds, y_bin, all_probs[:, crystals_idx]), f"model={model_path.name}, params={params}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["trivial", "baseline", "ann"], required=True)
    parser.add_argument("--splits-path", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/features_baseline"))
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.csv"))
    parser.add_argument("--baseline-model", type=Path, default=Path("experiments/models/baseline_lightgbm.joblib"))
    parser.add_argument("--ann-model", type=Path, default=Path("experiments/models/ann_initial.pt"))
    parser.add_argument("--ann-history", type=Path, default=Path("experiments/models/ann_initial_history.json"))
    parser.add_argument("--id", type=str, default=None, help="Log row id. Defaults to final_<model>_000.")
    return parser.parse_args()


def main() -> None:
    if not Path("docs/expectation.md").exists():
        raise RuntimeError(
            "docs/expectation.md must exist before final_eval.py runs (hard rule 8) - "
            "write the pre-registered expectation first."
        )

    args = parse_args()
    train_df = load_split(args.splits_path, "train")
    test_df = load_split(args.splits_path, "test")
    print(f"test set: {len(test_df)} images (read ONLY here, per hard rule 2)")

    if args.model == "trivial":
        metrics, detail = eval_trivial(train_df, test_df)
        model_label = "Trivial benchmark (always predict majority class) - TEST"
    elif args.model == "baseline":
        metrics, detail = eval_baseline(args.data_dir, args.cache_dir, args.baseline_model, test_df)
        model_label = "LightGBM (HOG+LBP) - TEST"
    else:
        metrics, detail = eval_ann(args.data_dir, args.ann_model, args.ann_history, test_df)
        model_label = "ANN Initial (frozen backbone) - TEST"

    print(f"\n{model_label}")
    print(f"  recall@P_MIN={metrics['recall_at_p_min']:.3f}")
    print(f"  crystals_pr_auc={metrics['crystals_pr_auc']:.3f}")
    print(f"  accuracy={metrics['accuracy']:.3f}")
    print(f"  confusion matrix (rows=true, cols=pred, order={CLASS_ORDER}): {metrics['confusion_matrix']}")

    log_row = {
        "id": args.id or f"final_{args.model}_000",
        "date": date.today().isoformat(),
        "model": model_label,
        "what_changed": f"final TEST-set evaluation (run once) - {detail}",
        "hyperparameters": json.dumps({}),
        "seed": SEED,
        "validation_metrics": json.dumps(metrics),
        "notes": "TEST SET result - scripts/final_eval.py",
    }
    pd.DataFrame([log_row]).to_csv(args.log_path, mode="a", header=False, index=False)
    print(f"\nlogged to {args.log_path}")


if __name__ == "__main__":
    main()
