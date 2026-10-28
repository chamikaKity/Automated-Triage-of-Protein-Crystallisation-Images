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

**Scope of the change:** regenerating moved more than just the directly-affected images - 6,328
of 14,367 images (44%) ended up in a different split than before, even though only 2,359 images
(16%) actually had their `group_key` change. `StratifiedGroupKFold`'s internal shuffling turned
out sensitive to the exact set of group labels, not just the group structure, so renaming
~2,000 already-singleton groups (from `single_<id>` to a precise `X<plate>_w<well>` key)
reshuffled many unrelated images too. Aggregate stats stayed stable (same split sizes, ~12.3%
Crystals in every split, both before and after), but this is exactly why the split must stay
fixed from here on - this single regeneration is the only one that will happen, and it happened
before any model touched the data. See `notebooks/01_eda.ipynb`'s "v1 vs v2 split comparison"
section for the full before/after quantification, and `data/splits_v1.csv` (kept only for this
comparison, not read anywhere else) plus git history for `scripts/make_splits.py` for the exact
diff.
