# MARCO Crystallisation Image Triage — Report

*Individual MSc Data Science AI coursework. Code: this repository. Experiment log:
`experiments/log.csv`. AI-use declaration: `docs/ai_use_log.md`.*

---

# Q1 — Dataset Description and EDA

*Source: `notebooks/01_eda.ipynb`, `scripts/download_extract.py`, `scripts/make_splits.py`.
Figures in `docs/figures/`.*

## Dataset description and licence

The dataset is MARCO (MAchine Recognition of Crystallization Outcomes), published by the
University at Buffalo Center for Computational Research (https://marco.ccr.buffalo.edu) under a
Creative Commons Attribution 4.0 International licence (CC BY 4.0). MARCO combines protein
crystallisation drop images from five contributing institutions, each expert-scored into one of
four classes: `Clear`, `Crystals`, `Precipitate`, `Other`. `Crystals` is the decision-relevant
class — it is the trigger for harvesting a drop and sending it for beamline data collection.

The full published dataset is distributed as TFRecord shards (~55GB, ~463,000 images; see
`scripts/download_extract.py` for the download/extract pipeline). For this coursework, 15 shards
were pulled (evenly spaced across the full shard range to avoid any shard-level sampling bias),
giving **14,367 images**. A fixed, one-time train/val/test split was then built
(`scripts/make_splits.py`) and is never regenerated — see the Leakage checks section below.

## Class and source distributions

All statistics in this section use **train + val only** (12,314 images); the test split
(2,053 images) is read only by `scripts/final_eval.py`, per the project's test-isolation rule.

**Class distribution** (Figure: `q1_class_distribution.png`):

| Class | Count | % |
|---|---|---|
| Precipitate | 5,893 | 47.9% |
| Clear | 4,195 | 34.1% |
| Crystals | 1,522 | 12.4% |
| Other | 704 | 5.7% |

Crystals is a genuine minority class at ~12% — consistent with the ~13% figure reported for the
published MARCO training set, confirming the 15-shard sample is representative.

**Source institution distribution** (Figure: `q1_source_distribution.png`): five source
institutions contribute images, very unevenly — source 3 alone accounts for 7,632 of 12,314
images (62%); sources 0 and 4 contribute only 371 and 225 images respectively.

**Class composition by source** (Figure: `q1_class_composition_by_source.png`):

| Source | Clear | Crystals | Other | Precipitate |
|---|---|---|---|---|
| 0 | 25.9% | **45.3%** | 8.4% | 20.5% |
| 1 | 40.3% | 25.5% | 3.5% | 30.7% |
| 2 | 33.1% | 20.0% | 8.0% | 38.8% |
| 3 | 33.2% | **4.9%** | 5.6% | 56.3% |
| 4 | 28.0% | 22.7% | 3.6% | 45.8% |

The Crystals rate swings nearly 9x across institutions (4.9% to 45.3%), and the single most
common institution (source 3, 62% of the data) has the *lowest* Crystals rate. See Finding 2
below — and Q3's error analysis, where this prediction is borne out on the test set.

**Image size and quality** (Figure: `q1_filesize_by_class.png`; dimensions sampled, n=500,
seed=42): image dimensions are not standardised — sizes range from 632×504 to 2448×2048, with
1280×960 as the single most common size (~70% of the sample) but far from universal. See
Finding 3 below.

## Leakage checks

Images of the same physical drop (same plate/well, imaged on different days or under different
microscopy modes) must stay entirely on one side of the split — otherwise a model could
effectively see a near-copy of a test image during training. MARCO's metadata carries no
explicit plate/well ID, so grouping keys were recovered from filename conventions, which differ
by source institution:

1. **`plateID<N>...wellNum<N>`** filenames → grouped by (plate, well) directly. 7,632 images
   (source 3).
2. **`<hash>_<well>_...`** filenames (e.g. `06d7_E4_CrossPolarization_9.jpg`) → grouped by
   (hash, well). 1,818 images (source 1).
3. **`X` + 9-digit plate + 4-digit well + 12-digit timestamp** filenames → grouped by (plate,
   well). 2,025 images (sources 0/2/4). This pattern was not obvious and was verified before use:
   the decoded well numbers fall in 1–1536 (a real high-density microplate format), and the
   decoded timestamps are all valid dates in 2004–2005, matching MARCO's known imaging period.
4. Remaining images with no filename grouping information (439 images): grouped by an 8×8
   average-image-hash, **restricted to same-label matches only** — a hash collision across
   *different* ground-truth labels is a false positive (confirmed empirically: one 51-image
   single-hash group turned out to be visually similar but physically different `Clear` wells,
   not one drop; see `q1_leakage_group_example.png`). Grouping unrelated images together this way
   is safe for the split (it never lets a true duplicate land on both sides), just conservative.

**A real leak was found and fixed.** The first version of this pipeline did not recognise
pattern 3 above, so those 2,025 images fell back to the hash method. Two near-identical
`Crystals` images of the same well (`X000005130_w1048`, imaged 3 weeks apart — same distinctive
star-burst crystal cluster, see `q1_xwell_leak_X000005130_w1048.png`) produced different hashes
(lighting changed between sessions) and so were never grouped — one landed in `test`, the other
in `val`. Once pattern 3 was added, the split was regenerated a single time, before any model had
used it (the project's "one fixed split" rule only permits a fixed split; this was treated as a
one-time methodology correction at the EDA stage, not a re-split for better numbers — see
`docs/split_v2_note.md`). Regenerating moved far more than the two affected images: 6,328 of
14,367 images (44%) ended up in a different split, because the splitter's internal shuffling
is sensitive to the exact set of group labels, not just the grouping structure — reinforcing why
the split is now permanently fixed.

**Comprehensive verification.** After the fix, every group across all four tiers was checked
against all three splits (not just train/val): **0 of 13,578 groups span more than one split.**
This is enforced both at build time (`scripts/make_splits.py` asserts it) and independently
re-verified in the EDA notebook.

**Label drift within groups.** Of 393 multi-image groups (train+val), 61 (15.5%) contain more
than one label. A deeper check cross-referencing the `_d<N>_` imaging-round marker in filenames
found 58 of these 61 drift groups are plate+well-pattern groups carrying day-number information,
and 47 of those 58 (81%) span more than one drop number — meaning most label drift is
**different drops sharing a well**, not one drop's state changing over time. The initial
"kinetic process" explanation was only partially right and was revised after checking. Either
way, the split still keeps the whole group together regardless of label; each group's majority
label is used only to decide which split it goes to, not as a claim the group has one true label.

**Limitation: relationships beyond the well.** All grouping above protects against the same
physical drop (same plate+well) appearing in more than one split. It does not, and cannot,
protect against the same protein or the same screening conditions being run on a *different*
plate — common practice in crystallisation screening (replicate plates, repeat attempts, broader
condition screens). Two images from different plates could still share protein-specific visual
characteristics without being the same physical drop. MARCO's metadata carries no protein ID,
batch ID, or experiment ID linking plates together, and nothing in the filenames identifies this
either, so this risk cannot be detected or controlled for with the metadata available. This is a
residual limitation of the dataset itself, not a gap in the grouping method described above.

## FP/FN consequences and metric justification

A false negative — a `Crystals` drop the model misses — can mean losing a hit produced from
scarce, expensive protein; the opportunity may not be recoverable. A false positive costs a
crystallographer a few minutes of unnecessary manual inspection. These costs are highly
asymmetric, and Crystals is a minority class (~12%), so **overall accuracy is unsuitable** as a
headline metric. Concretely: always predicting the majority class (`Precipitate`) scores 47.9%
accuracy while achieving 0% Crystals recall — every crystal hit would be missed despite a
seemingly reasonable accuracy figure.

**Primary metric:** recall on `Crystals` at a fixed minimum precision, **`P_MIN = 0.30`**, with
PR-AUC for Crystals as a supporting metric. Set from the classical baseline's validation PR
curve (base rate ~12.4%; `P_MIN = 0.30` gives ~85% recall while precision stays ~2.4x the base
rate — see `docs/primary_metric.md` for the full reasoning). This value is fixed for every model
family (classical baseline, every ANN variant, GenAI evaluation).

## Three findings that constrain modelling

1. **Crystals is a real minority class (~12% of train+val)**, and the cost of missing one is far
   higher than a false alarm. This rules out accuracy as the headline metric and means the loss
   function/sampling strategy needs to account for the imbalance directly (e.g. class weighting),
   not rely on the model discovering it unaided.
2. **Class mix varies sharply by source institution.** Source 3 (62% of all data) has only 4.9%
   Crystals, while source 0 has 45.3% Crystals — a 9x swing. Since one institution dominates the
   pool, a model could end up implicitly tuned to that institution's imaging style and
   under-perform on the others. This bears directly on the web app's stated limitation about
   unseen imagers (Q4) and is confirmed on the test set in Q3's error analysis.
3. **Image dimensions are not standardised.** A 500-image sample spans 632×504 up to 2448×2048,
   with 1280×960 as the dominant but not universal size (~70%). This constrains preprocessing —
   an explicit, documented resize/aspect-ratio strategy is needed rather than an assumption that
   every image already arrives at a fixed input size.

---

# Q2 — ANN Design

*Source: `src/marco/model.py`, `src/marco/data.py`, `docs/expectation.md`.*

## Architecture

A pretrained ResNet backbone (ResNet18 or ResNet34, ImageNet weights) with a new, deliberately
simple classifier head: `Dropout(0.4)` → `Linear(in_features, 4)`. Two fine-tuning degrees are
used across the project (see Q3): **frozen** (only the head trains — the Initial ANN) and
**layer4** (the backbone's last residual block also trains — the Final ANN).

![Architecture diagram](figures/q2_architecture.png)

*Figure: layer-by-layer shape and parameter counts (`scripts/make_architecture_diagram.py`,
read directly from the constructed model, not hand-copied). Blue = always frozen
(ImageNet-pretrained); orange = `layer4`, frozen in the Initial ANN and trainable in the Final
ANN; green = the head, always trainable.*

| | ResNet18 | ResNet34 |
|---|---|---|
| Total parameters | 11,178,564 | 21,286,724 |
| Trainable, frozen | 2,052 (head only) | 2,052 (head only) |
| Trainable, layer4 unfrozen | 8,395,780 | 13,116,420 |

Input: 224×224×3, direct resize (no centre crop — crystals can sit at the drop edge, so cropping
risks cutting out the decision-relevant signal), ImageNet mean/std normalisation.

## Justification of individual choices

- **Depth (ResNet18 vs ResNet34):** treated as the network-depth hyperparameter (one of the
  three required hyperparameters) rather than fixed in advance — compared empirically in the
  Q3 search, not assumed.
- **Width:** inherited from the pretrained backbone, not redesigned — modifying channel widths
  would discard the pretrained weights' structure for no clear benefit on a ~10K-image dataset.
- **Normalisation:** the backbone's existing BatchNorm layers are kept, but frozen (eval mode)
  when their parent conv layers are frozen too — `src/marco/model.py`'s `freeze_backbone_bn()`
  exists specifically because `model.train()` otherwise resets *every* submodule to train mode
  each epoch, including frozen ones, silently letting BatchNorm's running statistics drift even
  though the conv weights themselves are fixed.
- **Regularisation:** `Dropout(0.4)` in the head only. The learning curves (Q3) show no
  overfitting at all (val loss never diverges upward from train loss), so a heavier
  regularisation scheme was not pursued — dropout was kept fixed rather than swept as a 4th
  hyperparameter for exactly this reason (see `docs/ann_initial_training.md`).
- **Output layer / loss:** 4-way softmax via `CrossEntropyLoss`, **class-weighted** by
  inverse training-set frequency (`compute_class_weights()` in `src/marco/data.py`) — the same
  class-imbalance principle as the classical baseline's `class_weight='balanced'`, needed
  because of Q1's Finding 1 (Crystals is a genuine ~12% minority).
- **Head complexity:** a single Dropout+Linear layer, not a multi-layer MLP — the pretrained
  features are already strong, and a deeper head adds overfitting risk on a modest dataset for
  no established benefit.
- **Augmentation:** horizontal/vertical flip, 90°-multiple rotation (not continuous — a
  right-angle rotation needs no interpolation/padding), colour jitter, random greyscale, mild
  Gaussian blur — justified directly from the EDA (drop orientation carries no signal; source
  institutions differ in colour/quality, EDA Finding 2–3).

## Alternatives considered and rejected

- **Training from scratch:** rejected — ~10K images is too little to learn good low-level
  features from nothing; transfer learning from ImageNet is the standard, better-justified
  choice for a dataset this size.
- **A deeper/wider custom CNN:** rejected for the same reason as a complex head — no established
  benefit on this data scale, and harder to justify/explain at the viva than a well-known,
  explainable backbone.
- **Full fine-tuning (unfreezing the whole backbone) as the 4th hyperparameter:** considered but
  not pursued — higher overfitting risk on ~10K images, and the Initial ANN's learning curves
  already pointed specifically at `layer4` (not earlier layers) as the bottleneck, making a
  partial, targeted unfreeze the better-justified choice. Approved by the lecturer as the
  optional 4th hyperparameter — see `docs/lecturer_approval_fine_tuning.md`.
- **Dropout rate as a 5th swept hyperparameter:** rejected — the evidence (no observed
  overfitting) didn't support it, and CLAUDE.md only pre-names degree-of-fine-tuning as the
  sanctioned optional 4th. See `docs/ann_initial_training.md` for the full reasoning.

## Pre-registered expectation (Q2.b)

Written in full in `docs/expectation.md`, dated before `scripts/final_eval.py` ever touched the
test split, never edited afterwards. Summary: **the ANN was predicted to win on the primary
metric**, based on validation evidence (0.925 vs. 0.850 recall@P_MIN) and the expectation that
CNN features would capture Crystals-vs-Precipitate texture differences better than HOG+LBP — with
an explicit, honest caveat that validation PR-AUC actually favoured the baseline (0.652 vs.
0.572), so the prediction was not a foregone conclusion. See Q3 for whether this held on test.

---

# Q3 — Development, Optimisation, and Error Analysis

*Source: `scripts/train_ann.py`, `scripts/final_eval.py`, `experiments/log.csv`,
`notebooks/03_ann_initial.ipynb`, `docs/ann_initial_training.md`.*

## Hyperparameter search

Random search (seeded, `numpy.random.default_rng(42)`), 12 configs for the Initial ANN (frozen)
and 12 for the Final ANN (layer4), same grid for both so the only axis that differs between them
is the fine-tuning degree:

| Hyperparameter | Values | Required? |
|---|---|---|
| Learning rate | 0.0003, 0.001, 0.003, 0.01 | Required (1 of 3) |
| Batch size | 16, 32, 64 | Required (2 of 3) |
| Network depth | ResNet18, ResNet34 | Required (3 of 3) |
| Degree of fine-tuning | frozen, layer4 | Optional 4th, lecturer-approved |

This is 12 of the full 4×3×2=24-point grid per fine-tuning degree — the same order of magnitude
as the classical baseline's 24-config search (CLAUDE.md's fair-comparison rule). Optimiser:
Adam for all ANN configs; the classical baseline's gradient boosting (LightGBM) is the second,
mechanically different optimiser required by the coursework, compared directly on the same task.

**A real failed experiment** (one of the required ≥4): `ann_initial_010` (lr=0.01, batch=16)
diverged — loss rose to 2.85 instead of falling, recall collapsed to 0.622. High learning rate
combined with a small batch size destabilised training. Logged as-is in `experiments/log.csv`,
not re-run or hidden.

## Learning curves and under/overfitting diagnosis

Figure: `docs/figures/q3_ann_initial_learning_curves.png` (Initial ANN, best config). Both train
and val loss drop in epoch 1, then plateau — train loss sits around 0.99–1.01, val loss
fluctuates 0.85–0.97 for the remaining 11 epochs, with neither curve trending away from the
other. This rules out classic overfitting (val loss would climb while train loss kept falling)
but also means the frozen head stops improving almost immediately: with the backbone frozen,
only a Dropout+Linear head is learning, which is not enough capacity to keep extracting more
signal from the fixed ImageNet features. (Val loss sitting *below* train loss for most epochs is
expected, not a bug — `Dropout(0.4)` is active during training and off during evaluation.) This
plateau was the direct motivation for the Final ANN (layer4 unfreeze).

## Comparison table (test set, all 2,053 test images)

| Model | recall@P_MIN (primary) | Crystals PR-AUC | Accuracy | Training time | Inference latency |
|---|---|---|---|---|---|
| Trivial benchmark (majority class) | 0.000 | 0.124 | 0.480 | ~0s | ~0 ms |
| Classical baseline (LightGBM, HOG+LBP) | 0.937 | 0.678 | 0.812 | 28.9s | 0.03 ms (CPU) |
| Initial ANN (frozen) | 0.886 | 0.672 | 0.715 | 1060s | 6.3 ms (GPU) |
| **Final ANN (layer4 unfrozen)** | **0.965** | **0.823** | 0.809 | 1058s | 6.2 ms (GPU) |

**The pre-registered expectation (ANN wins) is confirmed — by the Final ANN.** But the result
is more nuanced than a clean win: **the Initial ANN actually loses to the baseline on test**
(0.886 vs. 0.937), despite winning clearly on validation (0.925 vs. 0.850) — a genuine
validation/test reversal, reported honestly rather than smoothed over (CLAUDE.md rule 7). The
Final ANN's advantage, by contrast, held and even grew relative to validation (0.965 on both),
which is stronger evidence that unfreezing `layer4` produced a real improvement rather than
overfitting to the validation set through repeated model selection.

One-time GPU training cost: Initial ANN ≈ $1.21, Final ANN ≈ $1.22 (RunPod RTX 4090, from
logged `train_time_s`) — see `docs/genai_evaluation.md` for the full cost discussion including
unproductive spend on unreliable rented GPU pods during this project (≈$3-5 more).

## Error analysis by class, source institution, and image quality

**By class** (Final ANN, test confusion matrix, rows=true, cols=predicted,
`experiments/log.csv` id `final_ann_final_000`):

```
                Clear  Crystals  Other  Precipitate
true Clear        602       52     17       30
true Crystals        5      219      8       22
true Other            4        9     91        9
true Precipitate     50       93     94      748
```

Precipitate is the largest source of confusion for every other class (e.g. 94 true-Other images
called Precipitate), consistent with Q1's observation that Precipitate and Crystals/Other can
look visually similar at the texture level.

**By source institution** (test set, Final ANN accuracy per source):

| Source | n (test) | Accuracy |
|---|---|---|
| 0 | 51 | 68.6% |
| 1 | 343 | 74.6% |
| 2 | 334 | 84.7% |
| 3 | 1,305 | 82.3% |
| 4 | 20 | 60.0% |

This directly confirms Q1's Finding 2: the two smallest, most underrepresented training sources
(0 and 4) have the lowest test accuracy (60-69%) despite source 0 having a *high* Crystals share
(55% of its test images) — the model simply hasn't seen enough examples from these imaging
setups to generalise as well, regardless of class mix. This is a concrete, measured instance of
the web app's stated "unseen imagers" limitation (Q4), not a hypothetical one.

**By image quality** (file size quartile as a resolution/detail proxy, same approach as the EDA):

| Quality quartile | n (test) | Accuracy |
|---|---|---|
| Q1 (smallest files) | 514 | 84.4% |
| Q2 | 513 | 82.5% |
| Q3 | 513 | 82.7% |
| Q4 (largest files) | 513 | 73.9% |

Counter-intuitively, the *largest* image files have the *lowest* accuracy, not the highest. A
plausible explanation is that file size correlates with source institution/imaging equipment
more strongly than with "quality" in the sense of model-helpfulness — larger files may come from
institutions whose imaging style is simply harder for this model (consistent with the
by-source table above), rather than large files being intrinsically harder to classify. This is
reported as an observed, somewhat surprising finding rather than over-interpreted causally.

---

# Q4 — Web Application

*Source: `app/backend/main.py`, `app/frontend/index.html`.*

A FastAPI backend serves a simple HTML/JS frontend. The model (the Final ANN, the best by the
primary metric) is loaded once at startup, but **every prediction runs a fresh forward pass on
the uploaded image** — nothing is precomputed or hard-coded (CLAUDE.md rule 10). The user
uploads an image, receives the predicted class, a per-class score bar chart, and a Grad-CAM
heatmap overlay (Q5).

**Explaining the score:** the response includes explicit text that the score is "the model's
output probability for each class, not a calibrated or guaranteed confidence... always inspect
flagged drops visually before acting on a prediction" — never presented as a guarantee.

**Stated limitations** (shown in the app itself): unseen microscopes/lighting/plate types may
reduce accuracy (directly informed by Q3's by-source error analysis), and the model has no way
to say "I don't recognise this" — demonstrated during testing (below).

**Testing:**

| Input | Result |
|---|---|
| Normal (real `Clear` test image) | Correctly predicted Clear, 88.9% |
| Invalid (non-image file) | Clean HTTP 400, "Could not read this file as an image" |
| Invalid (empty file) | Clean HTTP 400 |
| Invalid (missing upload field) | Clean HTTP 422 (FastAPI's standard validation error) |
| Out-of-distribution (random noise image) | **Confidently misclassified as Crystals, 56%** |

The out-of-distribution result is a genuine, useful finding, not a failure to hide: a standard
classifier has no mechanism to recognise "this doesn't look like any training class," and this
is now documented directly in the app's own limitations text, not just in this report.

---

# Q5 — Explainable AI

*Source: `src/marco/xai.py`, `app/backend/main.py`, `notebooks/04_xai.ipynb`,
`docs/references.md`.*

Three methods: **Grad-CAM** (architecture-specific, targets `layer4` — the block actually
fine-tuned, Selvaraju et al. 2017), live in the web app as the primary method; **Occlusion**
(model-agnostic, sliding-window sensitivity) and **LIME** (model-agnostic, superpixel-based,
Ribeiro et al. 2016), both compared against Grad-CAM in `notebooks/04_xai.ipynb`.

**Three incorrect test predictions analysed** (Figure: `docs/figures/q5_xai_comparison.png`):
a confident Crystals→Precipitate false negative (crystals_prob=0.008 — the costly error type
for this project), a confident Precipitate→Crystals false alarm (crystals_prob=0.624), and a
Crystals→Other confusion (crystals_prob=0.263). On the false negative, all three methods agree,
pointing at the drop's central region — the model is looking in a sensible place even though it
got the label wrong, suggesting the error is about feature *interpretation*, not attending to
irrelevant background. On the false alarm, occlusion genuinely **disagrees** with Grad-CAM/LIME
(its peak sensitivity is a small corner patch, not the drop centre) — reported honestly as a
real method disagreement rather than glossed over, since the three methods measure different
things (local gradient sensitivity vs. whole-block removal effect).

**Two required sanity checks:**
1. **Occlude the Grad-CAM region, check the prediction changes:** covering the highlighted
   region on the false-alarm case dropped the predicted Crystals probability from 0.624 to
   0.136 (Figure: `docs/figures/q5_sanity_occlusion.png`).
2. **Model randomisation** (Adebayo et al., 2018): re-initialising the trained layers
   (`layer4`+head) with random weights changed the Grad-CAM heatmap substantially — correlation
   -0.33 with the trained model's heatmap (Figure: `docs/figures/q5_sanity_randomization.png`),
   confirming the explanation genuinely depends on what the model learned, not just its
   architecture.

**Limitations, stated explicitly throughout** (notebook and app): none of these methods prove
the model's true internal reasoning. Agreement across methods is reassuring where it happens,
but a confounded feature correlating with the right region would produce the same agreement.
Grad-CAM is presented to the crystallographer as "where to look," not a verdict.

---

# Q6 — GenAI Evaluation

*Full detail in `docs/genai_evaluation.md`. Summary here.*

**Model:** `claude-haiku-4-5-20251001` (Anthropic Claude Haiku 4.5), via the Anthropic Messages
API, zero-shot (no MARCO training), 512px-resized images, JSON-only structured output
(`predicted_class` + `crystals_likelihood` 0-100, the latter used as the continuous score for
the primary metric). **Prompt selected on validation only** (12 images, never test): two
variants compared, v1 scored 6/12 vs. v2's 4/12 (v2's extra guidance made the model over-call
"Crystals"); v1 locked in before any test image was touched, both kept in
`experiments/prompts/` as the record.

**Evaluated on the full 2,053-image test set** (a smaller 250-image sample was run first to
measure real cost/time before committing to the full set — it was cheap enough, $0.16, that
there was no reason not to extend to the full set; both runs agree closely).

| Model | recall@P_MIN | Crystals PR-AUC | Accuracy | Latency | Cost/1000 | Training data |
|---|---|---|---|---|---|---|
| Final ANN (for reference) | 0.965 | 0.823 | 0.809 | 6.2 ms | $0 | ~10,261 images |
| **GenAI (Claude Haiku 4.5)** | **0.000** | 0.147 | 0.262 | 746 ms | $0.66, forever | **zero** |

**GenAI substantially underperforms every trained model, including the trivial majority-class
benchmark** — reported as-is per the honesty rule, not softened. It over-uses "Other" as a
catch-all (418 of 701 Clear images) and frequently confuses Precipitate with Crystals (365 of
985), consistent with a general-purpose model with no learned sense of what distinguishes these
four specific, visually subtle categories.

**Comparison:** GenAI needs zero labelled training images but costs money on every single
prediction, indefinitely, and is ~100x slower per prediction (network round-trip vs. a local
forward pass). It also sends every image to a third party over the network, with no offline
option — a material difference for any facility with data-sensitivity concerns, independent of
accuracy. The trained models cost a small one-time sum to train (~$1.21-1.22 each in GPU
rental) and then run free, fast, and fully offline.

**A realistic extension opportunity:** rather than replacing the trained classifier (which it
clearly cannot match here), a VLM could sit *downstream* of it — when the Final ANN flags a
drop as `Crystals`, send the image and its Grad-CAM overlay to a VLM and ask for a plain-language
description of what's visible in the highlighted region. This uses the VLM for what
general-purpose language models are actually good at (producing a readable summary for a
crystallographer) without asking it to do the specialised visual classification it performed
poorly at here.
