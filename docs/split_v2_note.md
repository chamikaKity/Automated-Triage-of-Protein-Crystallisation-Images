# Why data/splits.csv was regenerated once (2026-10-28)

The first version of `scripts/make_splits.py` grouped images by two filename patterns
(`plateID../wellNum..` and `<hash>_<well>_..`). Images with neither pattern (~2,857, sources 0/2/4)
fell back to an 8x8 average-image-hash, grouped conservatively by (hash, label).

EDA in `notebooks/01_eda.ipynb` found two things about that fallback:

1. A real leak: well `X000005130_w1048` has two near-identical `Crystals` images taken 3 weeks
   apart. Their average hashes differed (lighting/ring changed between sessions), so each became
   a one-image group with no grouping protection - one landed in `test`, the other in `val`.
2. A third filename pattern was missed: `X` + 9-digit plate + 4-digit well + 12-digit timestamp
   (`YYYYMMDDhhmm`) + `.png`. Confirmed genuine (not a spurious digit split) because the decoded
   well numbers fall in 1-1536, a real microplate density, and the decoded timestamps are all
   valid dates in 2004-2005, matching MARCO's known imaging period. This pattern covers 2,032 of
   the 2,857 fallback images with precise plate+well grouping instead of the weaker hash.

**Decision:** regenerate `data/splits.csv` once, now, before any model has used it - see
CLAUDE.md rule 1 ("never re-split"). This is a documented leakage-methodology fix made at the
EDA stage, not a re-split chosen for better metrics; no model selection, threshold, or evaluation
has touched the split yet. The split will not be regenerated again after this.

Old split (v1) is kept for comparison; see git history for `scripts/make_splits.py` and
`data/splits.csv` for the exact before/after diff.
