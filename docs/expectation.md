# Pre-registered expectation (Q2.b)

**Date: 2026-11-03.** Written before `scripts/final_eval.py` has ever been run - the test
split has not been touched. Per CLAUDE.md hard rule 8, this is never edited after this date,
regardless of what the test results turn out to be.

## Prediction

**The ANN (Initial ANN: frozen ResNet18 backbone + head) is expected to win on the primary
metric (recall on Crystals at P_MIN=0.30) on the test set**, over the classical baseline
(LightGBM on HOG+LBP features).

## Reasoning

1. **Validation evidence already shows a real, non-borderline gap.** Initial ANN:
   recall@P_MIN=0.925. Classical baseline: recall@P_MIN=0.850. A 7.5-point margin on 2,053
   validation images is large enough that it's unlikely to be pure noise, though test could
   still differ (see caveat below).
2. **Richer features for a visually subtle class.** Crystal formations vary continuously in
   shape, size and arrangement; ImageNet-pretrained CNN features capture spatial/textural
   patterns that HOG+LBP's fixed edge-orientation histograms are less suited to, especially
   for Crystals vs. Precipitate, the two classes that visually overlap most (see the
   confusion matrix in `notebooks/03_ann_initial.ipynb`).
3. **The comparison is fair, not stacked.** Both models use the same fixed split
   (`data/splits.csv`), the same class-weighted-loss principle, and tuning budgets of the
   same order of magnitude (24 LightGBM configs vs. 12 ANN configs) - the baseline wasn't
   under-tuned relative to the ANN.

## Caveat - the honest uncertainty here

Crystals PR-AUC actually favours the **baseline** on validation (0.652 vs. the ANN's 0.572).
The ANN's advantage is concentrated specifically at the `P_MIN=0.30` operating point, not a
blanket superiority across every precision level. If the test set's score distribution shifts
even slightly, this could plausibly flip which model wins on the primary metric - this is a
genuine prediction, not a foregone conclusion dressed up as one.

If the test result contradicts this prediction, that is reported as-is per CLAUDE.md rule 7
(never weaken or hide a result that doesn't match expectations).
