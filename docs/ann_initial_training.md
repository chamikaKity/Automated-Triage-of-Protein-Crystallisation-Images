# Initial ANN training

What we trained, how, and what came out of it. Covers `src/marco/model.py`,
`src/marco/data.py`, `scripts/train_ann.py`, and the results in
`notebooks/03_ann_initial.ipynb`.

## What we trained

A pretrained ResNet (18 or 34, ImageNet weights) with its backbone **frozen** and a new head
(`Dropout(0.4)` + `Linear`) trained on top, for the 4-class problem (`Clear`, `Crystals`,
`Other`, `Precipitate`). This is the "Initial ANN" - the first working CNN, before any
unfreezing. Only 2,052 parameters actually train (the head); the backbone's ~11-21M parameters
stay fixed at their ImageNet values.

## Training setup

| Setting | Value | Why |
|---|---|---|
| Data | `data/splits.csv` - train (10,261) / val (2,053) | Same fixed split as everything else; test untouched |
| Loss | Class-weighted `CrossEntropyLoss` | Inverse-frequency weights, same principle as the baseline's `class_weight='balanced'` |
| Optimiser | Adam | Every config here uses Adam; the required second optimiser for comparison hasn't been run yet |
| Epochs | 12 per config | Fixed across all configs, for a fair comparison |
| Image size | 224x224, direct resize (no crop) | Crystals can sit at the drop edge - cropping risks cutting the signal out |
| Augmentation | Horizontal/vertical flip, 90-degree rotation, colour jitter, random greyscale, mild Gaussian blur | Drop orientation carries no signal; source institutions differ in colour/quality (EDA findings 2-3) |
| Seed | 42 | Fixed and recorded |

## Hyperparameter categories searched

Three required categories from CLAUDE.md's fixed list (learning rate, batch size, network
depth). Values were sampled randomly from this grid, not grid-searched exhaustively:

| Category | Values tried | What it resolves |
|---|---|---|
| Learning rate | 0.0003, 0.001, 0.003, 0.01 | Required hyperparameter 1 |
| Batch size | 16, 32, 64 | Required hyperparameter 2 |
| Network depth | ResNet18, ResNet34 | Required hyperparameter 3 - backbone depth stands in for "network depth" since the head itself is a single layer |

Two things were deliberately **not** swept here, both pending the lecturer's sign-off:

| Not yet swept | Status |
|---|---|
| Dropout rate (fixed at 0.4) | Would be a 4th hyperparameter beyond the required three - needs approval |
| Degree of fine-tuning (frozen vs. partial unfreeze) | This run is frozen-only; partial unfreeze is the planned "Final ANN" - needs approval |

## Search method

12 configs; each independently samples a backbone, learning rate, and batch size
(`numpy.random.default_rng(seed=42)`) from the 4x3x2 = 24-point grid above, not 12 unique
points drawn without replacement - in practice this gave 11 unique combinations with one
repeat (`resnet34`, lr=0.001, batch=32, tested twice). Same order of magnitude as the classical
baseline's 24-config search (CLAUDE.md's fair-comparison rule). Each config trains for all 12
epochs; the config
with the best `recall@P_MIN` (the primary metric itself, not a proxy) is kept.

## Results

| id | backbone | lr | batch | recall@P_MIN | Crystals PR-AUC | train time |
|---|---|---|---|---|---|---|
| ann_initial_000 | resnet18 | 0.0003 | 32 | 0.917 | 0.569 | 1115s |
| ann_initial_001 | resnet18 | 0.003 | 64 | 0.921 | 0.593 | 1095s |
| **ann_initial_002** | **resnet18** | **0.001** | **16** | **0.925** | **0.572** | **1060s** |
| ann_initial_003 | resnet18 | 0.001 | 64 | 0.898 | 0.548 | 1098s |
| ann_initial_004 | resnet34 | 0.0003 | 64 | 0.886 | 0.559 | 1071s |
| ann_initial_005 | resnet34 | 0.001 | 16 | 0.878 | 0.565 | 1043s |
| ann_initial_006 | resnet34 | 0.003 | 32 | 0.870 | 0.560 | 1046s |
| ann_initial_007 | resnet18 | 0.01 | 64 | 0.862 | 0.566 | 1076s |
| ann_initial_008 | resnet34 | 0.001 | 32 | 0.890 | 0.560 | 1049s |
| ann_initial_009 | resnet34 | 0.001 | 32 | 0.909 | 0.551 | 1104s |
| ann_initial_010 | resnet18 | 0.01 | 16 | **0.622** | 0.390 | 1037s |
| ann_initial_011 | resnet34 | 0.0003 | 16 | 0.894 | 0.560 | 1058s |

**Best: `ann_initial_002`** - resnet18, lr=0.001, batch=16 -> recall@P_MIN=0.925.

**Failed experiment:** `ann_initial_010` (lr=0.01, batch=16) diverged - loss rose to 2.85
instead of falling, recall collapsed to 0.622. High learning rate combined with a small batch
size destabilised training. Logged as-is, not re-run or hidden - one of the required
unsuccessful experiments.

## vs. the classical baseline

| model | recall@P_MIN | Crystals PR-AUC | train time | inference latency |
|---|---|---|---|---|
| Classical baseline (LightGBM, HOG+LBP) | 0.850 | 0.652 | 28.9s | 0.03 ms/image |
| Initial ANN (resnet18, frozen) | **0.925** | 0.572 | 1060s | 6.34 ms/image |

The ANN wins on the primary metric despite a lower PR-AUC - it reaches higher recall at the
fixed `P_MIN=0.30` precision floor, which is what actually matters for the triage decision.
Training time and inference latency are both much higher, and were measured on different
hardware (baseline: this laptop's CPU; ANN: GPU), so that gap isn't a clean comparison yet.

## Where everything lives

- `experiments/log.csv` - rows `ann_initial_000` to `ann_initial_011`
- `experiments/models/ann_initial.pt`, `ann_initial_history.json` - best model + its per-epoch history (gitignored, kept locally)
- `notebooks/03_ann_initial.ipynb` - figures and full analysis (learning curves, confusion matrix, PR curve)

## Infrastructure note

Frozen-backbone training still needs a GPU to run in reasonable time - a CPU run of this same
search would take far longer. It was run on a RunPod GPU pod (RTX 4090) after running into GPU
quota limits on Colab. All 12 configs ran sequentially on a single pod: summing each config's
logged `train_time_s` in `experiments/log.csv` gives **~3.57 hours of GPU time (~$1.21 at
$0.34/hr)**, plus some additional wall-clock time for setup and the one-off dataset download.
The code itself (`scripts/train_ann.py`) is unchanged by where it runs. For the next search
(the Final ANN), the plan is to split the config list across several pods running concurrently
to cut wall-clock time without changing the total compute used - see `scripts/train_ann.py`'s
`--start-idx`/`--end-idx` options once added.
