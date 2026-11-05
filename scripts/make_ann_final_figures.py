"""Build the Final ANN's hyperparameter-search and learning-curve figures for the report.

Mirrors notebooks/03_ann_initial.ipynb's equivalent cells exactly (same style/colors), but for
the Final ANN (layer4+head unfrozen) - those two figures existed for the Initial ANN but were
never generated for the Final ANN. Reads only experiments/log.csv and the saved
ann_final_history.json; no training happens here.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 100


def main() -> None:
    log = pd.read_csv(PROJECT_ROOT / "experiments" / "log.csv")
    # id-prefix filter, not model-name startswith: the latter also matches the official TEST-set
    # row ("ANN Final (layer4+head unfrozen) - TEST"), silently adding a 13th, non-validation bar.
    ann_runs = log[log["id"].str.startswith("ann_final_")].reset_index(drop=True)
    ann_metrics = ann_runs["validation_metrics"].apply(json.loads)
    ann_runs["recall_at_p_min"] = ann_metrics.apply(lambda m: m["recall_at_p_min"])
    print(f"Final ANN configs found: {len(ann_runs)}, ids={ann_runs['id'].tolist()}")

    best_idx = ann_runs["recall_at_p_min"].idxmax()
    colors = ["#D55E00" if i == best_idx else "#999999" for i in range(len(ann_runs))]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(ann_runs)), ann_runs["recall_at_p_min"], color=colors, width=0.7)
    ax.set_xlabel("config index (0-11)")
    ax.set_ylabel("val recall @ P_MIN=0.30")
    ax.set_title("Final ANN: 12-config hyperparameter search, layer4 unfrozen (best config in orange)")
    fig.savefig(FIGURES_DIR / "q3_ann_final_search.png", dpi=150, bbox_inches="tight")
    print(f"saved {FIGURES_DIR / 'q3_ann_final_search.png'}")
    print(f"best: {ann_runs.loc[best_idx, 'id']}, recall@P_MIN={ann_runs.loc[best_idx, 'recall_at_p_min']:.3f}")

    ckpt = json.load(open(PROJECT_ROOT / "experiments" / "models" / "ann_final_history.json"))
    params, history = ckpt["params"], ckpt["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], color="#0072B2", lw=2, marker="o", ms=4, label="train")
    axes[0].plot(epochs, history["val_loss"], color="#D55E00", lw=2, marker="o", ms=4, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("class-weighted cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_crystals_pr_auc"], color="#D55E00", lw=2, marker="o", ms=4)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val Crystals PR-AUC")
    axes[1].set_title("Validation Crystals PR-AUC")

    fig.suptitle(f"Final ANN best config: {params['backbone']}, lr={params['learning_rate']}, "
                 f"batch={params['batch_size']}, layer4 unfrozen")
    fig.savefig(FIGURES_DIR / "q3_ann_final_learning_curves.png", dpi=150, bbox_inches="tight")
    print(f"saved {FIGURES_DIR / 'q3_ann_final_learning_curves.png'}")


if __name__ == "__main__":
    main()
