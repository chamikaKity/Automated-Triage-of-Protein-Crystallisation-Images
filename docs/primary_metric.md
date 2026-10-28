# Primary metric: P_MIN = 0.30 (set 2026-10-28)

**Primary metric:** recall on `Crystals` at a fixed minimum precision, `P_MIN = 0.30`. Set once,
using **validation data only** from the classical baseline's PR curve (`baseline_lgbm_022` in
`experiments/log.csv` - the best of the 24-config LightGBM search, val `crystals_pr_auc=0.652`).
Test data was not touched. This value is now fixed and used for every model family (classical
baseline, every ANN variant, GenAI evaluation) for a fair comparison.

## Reasoning

Crystals base rate in validation is ~12.4% (matches the EDA's train+val figure), so any
precision meaningfully above that reflects real discriminative skill, not chance.

| Precision | Recall | Enrichment over base rate |
|---|---|---|
| 0.25 | 0.90 | ~2.0x |
| **0.30** | **0.85** | **~2.4x** |
| 0.35 | 0.80 | ~2.8x |
| 0.45 | 0.74 | ~3.6x |
| 0.54 | 0.67 | ~4.3x |
| 0.67 | 0.56 | ~5.4x |

The project's own cost framing favours recall: a missed crystal (false negative) can lose a hit
from scarce, expensive protein, while a false positive costs a crystallographer only a few
minutes of inspection - and the system never discards drops automatically, only prioritises them.
`P_MIN = 0.30` keeps recall high (~85%) while precision is still ~2.4x the base rate, so a
flagged drop is a meaningfully stronger signal than chance, not just "flag everything."

## How to apply

- `recall@P_MIN=0.30` on Crystals is the number reported in every model comparison (Q3's
  trivial-benchmark/classical-baseline/initial-ANN/final-ANN table, and the Q6 GenAI comparison).
- PR-AUC for Crystals remains the supporting, threshold-independent metric.
- This file is the record required by CLAUDE.md's metrics section; the "P_MIN = TBD" placeholder
  there has been updated to reference this decision.
