"""Train the Initial or Final ANN (ResNet18/34 + head) and log every run.

Hyperparameter search over the three required dimensions (learning rate, batch size,
network depth = resnet18 vs resnet34), at a fixed fine-tuning degree. --fine-tune-degree
frozen (default) trains the Initial ANN (only the head trains); layer4 trains the Final ANN
(the backbone's last block also trains) - the optional 4th hyperparameter, approved by the
lecturer, see docs/lecturer_approval_fine_tuning.md.

Model selection uses recall at P_MIN=0.30 directly (not a proxy like PR-AUC), since P_MIN is
already fixed - see docs/primary_metric.md.

Needs a GPU to run the full search in reasonable time; CPU is fine only for a tiny smoke test.

--start-idx/--end-idx let the same fixed, seeded config list be split across multiple GPUs
running concurrently (each given a different slice) to cut wall-clock time without changing
which configs get run - see docs/ann_initial_training.md's infrastructure note.
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
from marco.model import FINE_TUNE_DEGREES, build_model, count_parameters, freeze_backbone_bn

SEED = 42
EPOCHS = 12
N_CONFIGS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, fine_tune_degree: str) -> float:
    model.train()
    freeze_backbone_bn(model, fine_tune_degree)
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
    parser.add_argument("--fine-tune-degree", choices=FINE_TUNE_DEGREES, default="frozen",
                         help="frozen = Initial ANN (only the head trains); layer4 = Final ANN "
                              "(backbone's last block also trains) - pending lecturer approval.")
    parser.add_argument("--model-out", type=Path, default=None,
                         help="Defaults to experiments/models/ann_initial.pt or ann_final.pt, "
                              "based on --fine-tune-degree.")
    parser.add_argument("--history-out", type=Path, default=None,
                         help="Defaults alongside --model-out.")
    parser.add_argument("--n-configs", type=int, default=N_CONFIGS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--start-idx", type=int, default=0,
                         help="First config index to run (inclusive) - for splitting the search across GPUs.")
    parser.add_argument("--end-idx", type=int, default=None,
                         help="Last config index to run (exclusive). Defaults to --n-configs (run everything).")
    args = parser.parse_args()
    if args.end_idx is None:
        args.end_idx = args.n_configs
    if not 0 <= args.start_idx < args.end_idx <= args.n_configs:
        raise ValueError(f"need 0 <= start_idx < end_idx <= n_configs, got "
                          f"start_idx={args.start_idx}, end_idx={args.end_idx}, n_configs={args.n_configs}")

    run_name = "ann_initial" if args.fine_tune_degree == "frozen" else "ann_final"
    if args.model_out is None:
        args.model_out = Path(f"experiments/models/{run_name}.pt")
    if args.history_out is None:
        args.history_out = Path(f"experiments/models/{run_name}_history.json")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(f"device: {DEVICE}")

    train_df = load_split(args.splits_path, "train")
    val_df = load_split(args.splits_path, "val")
    class_weights = compute_class_weights(train_df).to(DEVICE)

    val_loader = DataLoader(MarcoDataset(val_df, args.data_dir, train=False),
                             batch_size=64, shuffle=False, num_workers=0)

    run_name = "ann_initial" if args.fine_tune_degree == "frozen" else "ann_final"
    model_label = "ANN Initial" if args.fine_tune_degree == "frozen" else "ANN Final"
    fine_tune_desc = "frozen backbone" if args.fine_tune_degree == "frozen" else "layer4+head unfrozen"

    rng = np.random.default_rng(args.seed)
    all_params = [sample_hyperparams(rng) for _ in range(args.n_configs)]
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    best = None

    for i in range(args.start_idx, args.end_idx):
        params = all_params[i]
        train_loader = DataLoader(MarcoDataset(train_df, args.data_dir, train=True),
                                   batch_size=params["batch_size"], shuffle=True, num_workers=0)

        model = build_model(params["backbone"], fine_tune_degree=args.fine_tune_degree).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=params["learning_rate"]
        )

        history = {"train_loss": [], "val_loss": [], "val_crystals_pr_auc": []}
        t0 = time.time()
        for epoch in range(args.epochs):
            epoch_t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.fine_tune_degree)
            epoch_val = evaluate(model, val_loader, criterion)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(epoch_val["loss"])
            history["val_crystals_pr_auc"].append(epoch_val["crystals_pr_auc"])
            print(f"  config {i + 1}/{args.n_configs} epoch {epoch + 1}/{args.epochs}: "
                  f"train_loss={train_loss:.3f} val_loss={epoch_val['loss']:.3f} "
                  f"val_crystals_pr_auc={epoch_val['crystals_pr_auc']:.3f} "
                  f"({time.time() - epoch_t0:.1f}s)")
        train_time_s = time.time() - t0

        t0 = time.time()
        final_metrics = evaluate(model, val_loader, criterion)
        infer_latency_ms = (time.time() - t0) / len(val_loader.dataset) * 1000

        print(f"[{i + 1}/{args.n_configs}] {params} -> "
              f"recall@P_MIN={final_metrics['recall_at_p_min']:.3f} "
              f"crystals_pr_auc={final_metrics['crystals_pr_auc']:.3f} "
              f"accuracy={final_metrics['accuracy']:.3f}")

        log_row = {
            "id": f"{run_name}_{i:03d}",
            "date": date.today().isoformat(),
            "model": f"{model_label} ({params['backbone']}, {fine_tune_desc})",
            "what_changed": f"hyperparameter search config {i + 1}/{args.n_configs}",
            "hyperparameters": json.dumps(params),
            "seed": args.seed,
            "validation_metrics": json.dumps({
                **final_metrics,
                "train_time_s": round(train_time_s, 2),
                "inference_latency_ms_per_image": round(infer_latency_ms, 4),
            }),
            "notes": "",
        }
        pd.DataFrame([log_row]).to_csv(args.log_path, mode="a", header=False, index=False)

        if best is None or final_metrics["recall_at_p_min"] > best["metrics"]["recall_at_p_min"]:
            best = {"model": model, "params": params, "metrics": final_metrics, "history": history}
            torch.save(best["model"].state_dict(), args.model_out)
            args.history_out.write_text(
                json.dumps({"params": best["params"], "history": best["history"]}, indent=2)
            )

    trainable, total = count_parameters(best["model"])
    print(f"\nbest config in this run's slice [{args.start_idx}:{args.end_idx}] "
          f"saved to {args.model_out}: {best['params']}")
    print(f"params: {trainable:,} trainable / {total:,} total")
    print(f"best metrics: {best['metrics']}")


if __name__ == "__main__":
    main()
