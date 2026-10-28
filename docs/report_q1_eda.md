# Q1 — Dataset Description and EDA

*Draft for the written report. Source: `notebooks/01_eda.ipynb`, `scripts/download_extract.py`,
`scripts/make_splits.py`. Figures referenced below are in `docs/figures/`.*

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
below.

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

**Primary metric:** recall on `Crystals` at a fixed minimum precision (`P_MIN`), with PR-AUC for
Crystals as a supporting metric. `P_MIN` itself is still to be set — doing so needs an actual
validation PR curve from a trained model, which does not exist yet at the EDA stage; it will be
fixed once the classical baseline produces a first PR curve, using validation data only.

## Three findings that constrain modelling

1. **Crystals is a real minority class (~12% of train+val)**, and the cost of missing one is far
   higher than a false alarm. This rules out accuracy as the headline metric and means the loss
   function/sampling strategy needs to account for the imbalance directly (e.g. class weighting),
   not rely on the model discovering it unaided.
2. **Class mix varies sharply by source institution.** Source 3 (62% of all data) has only 4.9%
   Crystals, while source 0 has 45.3% Crystals — a 9x swing. Since one institution dominates the
   pool, a model could end up implicitly tuned to that institution's imaging style and
   under-perform on the others. This bears directly on the web app's stated limitation about
   unseen imagers (Q4) and suggests checking per-source performance during error analysis (Q3),
   not just aggregate metrics.
3. **Image dimensions are not standardised.** A 500-image sample spans 632×504 up to 2448×2048,
   with 1280×960 as the dominant but not universal size (~70%). This constrains preprocessing —
   an explicit, documented resize/aspect-ratio strategy is needed rather than an assumption that
   every image already arrives at a fixed input size.
