"""Train the Initial ANN (frozen ResNet18/34 backbone + head) and log every run.

Hyperparameter search over the three required dimensions (learning rate, batch size,
network depth = resnet18 vs resnet34). Fine-tuning stays frozen throughout - the Final ANN
(partial unfreeze) is a separate, later step pending lecturer approval, see docs/.

Model selection uses recall at P_MIN=0.30 directly (not a proxy like PR-AUC), since P_MIN is
already fixed - see docs/primary_metric.md.

Needs a GPU to run the full search in reasonable time; CPU is fine only for a tiny smoke test.
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from marco.data import CLASS_ORDER, MarcoDataset, compute_class_weights, load_split
from marco.metrics import recall_at_precision
from marco.model import build_model, count_parameters, freeze_backbone_bn

SEED = 42
EPOCHS = 12
N_CONFIGS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    freeze_backbone_bn(model)
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        total_loss += criterion(outputs, labels).item() * imgs.size(0)
        all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    crystals_idx = CLASS_ORDER.index("Crystals")
    y_bin = (all_labels == crystals_idx).astype(int)
    crystals_scores = all_probs[:, crystals_idx]

    return {
        "loss": total_loss / len(loader.dataset),
        "crystals_pr_auc": float(average_precision_score(y_bin, crystals_scores)),
        "recall_at_p_min": recall_at_precision(y_bin, crystals_scores),
        "accuracy": float((all_probs.argmax(axis=1) == all_labels).mean()),
    }


def sample_hyperparams(rng: np.random.Generator) -> dict:
    return {
        "backbone": str(rng.choice(["resnet18", "resnet34"])),
        "learning_rate": float(rng.choice([0.01, 0.003, 0.001, 0.0003])),
        "batch_size": int(rng.choice([16, 32, 64])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("experiments/models/ann_initial.pt"))
    parser.add_argument("--history-out", type=Path,
                         default=Path("experiments/models/ann_initial_history.json"))
    parser.add_argument("--n-configs", type=int, default=N_CONFIGS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(f"device: {DEVICE}")

    train_df = load_split(args.splits_path, "train")
    val_df = load_split(args.splits_path, "val")
    class_weights = compute_class_weights(train_df).to(DEVICE)

    val_loader = DataLoader(MarcoDataset(val_df, args.data_dir, train=False),
                             batch_size=64, shuffle=False, num_workers=2)

    rng = np.random.default_rng(args.seed)
    log_rows = []
    best = None

    for i in range(args.n_configs):
        params = sample_hyperparams(rng)
        train_loader = DataLoader(MarcoDataset(train_df, args.data_dir, train=True),
                                   batch_size=params["batch_size"], shuffle=True, num_workers=2)

        model = build_model(params["backbone"]).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=params["learning_rate"]
        )

        history = {"train_loss": [], "val_loss": [], "val_crystals_pr_auc": []}
        t0 = time.time()
        for _ in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            epoch_val = evaluate(model, val_loader, criterion)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(epoch_val["loss"])
            history["val_crystals_pr_auc"].append(epoch_val["crystals_pr_auc"])
        train_time_s = time.time() - t0

        t0 = time.time()
        final_metrics = evaluate(model, val_loader, criterion)
        infer_latency_ms = (time.time() - t0) / len(val_loader.dataset) * 1000

        print(f"[{i + 1}/{args.n_configs}] {params} -> "
              f"recall@P_MIN={final_metrics['recall_at_p_min']:.3f} "
              f"crystals_pr_auc={final_metrics['crystals_pr_auc']:.3f} "
              f"accuracy={final_metrics['accuracy']:.3f}")

        log_rows.append({
            "id": f"ann_initial_{i:03d}",
            "date": date.today().isoformat(),
            "model": f"ANN Initial ({params['backbone']}, frozen backbone)",
            "what_changed": f"hyperparameter search config {i + 1}/{args.n_configs}",
            "hyperparameters": json.dumps(params),
            "seed": args.seed,
            "validation_metrics": json.dumps({
                **final_metrics,
                "train_time_s": round(train_time_s, 2),
                "inference_latency_ms_per_image": round(infer_latency_ms, 4),
            }),
            "notes": "",
        })

        if best is None or final_metrics["recall_at_p_min"] > best["metrics"]["recall_at_p_min"]:
            best = {"model": model, "params": params, "metrics": final_metrics, "history": history}

    pd.DataFrame(log_rows).to_csv(args.log_path, mode="a", header=False, index=False)
    print(f"\nlogged {len(log_rows)} configs to {args.log_path}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["model"].state_dict(), args.model_out)
    args.history_out.write_text(json.dumps({"params": best["params"], "history": best["history"]}, indent=2))

    trainable, total = count_parameters(best["model"])
    print(f"best config saved to {args.model_out}: {best['params']}")
    print(f"params: {trainable:,} trainable / {total:,} total")
    print(f"best metrics: {best['metrics']}")


if __name__ == "__main__":
    main()
