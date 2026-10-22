# Automated Triage of Protein Crystallisation Images

## Data split

We downloaded 15 shards of the MARCO dataset (~14,400 images, 4 classes: Clear, Crystals,
Precipitate, Other).

The images were split once into:
- **Train** – 71% (~10,260 images)
- **Validation** – 14% (~2,050 images)
- **Test** – 14% (~2,050 images)

This split is saved in `data/splits.csv` and is never changed or re-run. All models (baseline,
ANN, GenAI) use the exact same split so results can be fairly compared.

Images from the same plate/well are always kept in the same split, so the model can't "cheat"
by seeing the same drop in both training and testing.
