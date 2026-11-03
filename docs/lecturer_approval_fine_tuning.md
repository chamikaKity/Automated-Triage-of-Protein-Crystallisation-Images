# Lecturer approval: degree of fine-tuning as the optional 4th hyperparameter

**Date: 2026-11-03.** Dr. Rajadurai approved adding degree of fine-tuning (frozen backbone
vs. layer4+head unfrozen) as the optional 4th hyperparameter, alongside the required three
(learning rate, batch size, network depth).

Scope, as discussed:
- Two levels only: `frozen` (Initial ANN, already trained) and `layer4` (Final ANN, backbone's
  last residual block also trains). Full fine-tuning was not pursued - see
  `docs/ann_initial_training.md` for why (overfitting risk on ~10K training images, and
  `layer4` is already well-motivated by the Initial ANN's plateaued learning curves).
- Dropout rate stays fixed at 0.4, not separately swept - see the reasoning recorded when this
  was discussed (dropout is unlikely to matter given no overfitting was observed, and CLAUDE.md
  only pre-names fine-tuning degree as the sanctioned optional 4th).

This unblocks `scripts/train_ann.py --fine-tune-degree layer4`, which was already built and
code-tested (see `src/marco/model.py`, commit `2016e35`) but not run for real until this
approval.
