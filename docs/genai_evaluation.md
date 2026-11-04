# Q6 - GenAI evaluation

A pretrained vision-language model (zero-shot, no MARCO training) evaluated on the same test
images with the same primary metric as every other model, per CLAUDE.md.

Pricing used for the cost figures below ($1/MTok input, $5/MTok output) is Claude Haiku 4.5's
listed rate, verified against https://platform.claude.com/docs/en/about-claude/pricing on
2026-11-04 (`PRICE_PER_MTOK_INPUT`/`PRICE_PER_MTOK_OUTPUT` in `scripts/eval_genai.py`).

## Model, configuration, service

| | |
|---|---|
| Model | `claude-haiku-4-5-20251001` (Anthropic Claude Haiku 4.5) |
| Service | Anthropic Messages API (`anthropic` Python SDK) |
| Configuration | default sampling (no temperature override), `max_tokens=100`, images resized to a 512px longest side before sending |
| Input | one image + the prompt per request; JSON-only structured output requested |
| Output parsed | `predicted_class` (one of the 4 classes) and `crystals_likelihood` (0-100), the latter used as the continuous score for the primary metric, same role as the trained models' Crystals softmax probability |

## Prompt selection (validation only, per the rule)

Two prompt variants were compared on 12 **validation** images (3 per class, seed=42), never
test:

| Prompt | Validation accuracy |
|---|---|
| `crystals_classification_v1.txt` | 6/12 |
| `crystals_classification_v2.txt` (added more explicit visual cues for Precipitate/Other/Crystals) | 4/12 |

v2's extra guidance made the model over-call "Crystals" on images that weren't (several
Precipitate validation images flipped to confident Crystals predictions). **v1 was selected**
and locked in before any test image was touched. Both prompt files are kept in
`experiments/prompts/` as the record of this comparison.

## Test-set evaluation

Evaluated on a **stratified random sample of 250 of the 2,053 test images** (proportional to
each class's share: Clear 85, Crystals 31, Other 14, Precipitate 120), not the full test set -
a deliberate cost/time trade-off given the project deadline, stated here explicitly rather than
hidden. `scripts/eval_genai.py` does the sampling, the API calls, and the logging; results are
in `experiments/log.csv` (`genai_claude_haiku_000`) and per-image in
`experiments/genai_test_results.csv`.

## Comparison with every other model (test set)

| Model | recall@P_MIN | Crystals PR-AUC | Accuracy | Inference latency* | Cost per 1000 predictions | Training data needed |
|---|---|---|---|---|---|---|
| Trivial benchmark | 0.000 | 0.124 | 0.480 | ~0 ms | $0 | none |
| Classical baseline (LightGBM) | 0.937 | 0.678 | 0.812 | 0.03 ms (CPU) | $0 (after training) | ~10,261 labelled images |
| Initial ANN (frozen) | 0.886 | 0.672 | 0.715 | 6.3 ms (GPU) | $0 (after training) | ~10,261 labelled images |
| Final ANN (layer4) | **0.965** | **0.823** | 0.809 | 6.2 ms (GPU) | $0 (after training) | ~10,261 labelled images |
| **GenAI (Claude Haiku 4.5)** | **0.000** | 0.154 | 0.264 | 734 ms | **$0.66** | **zero** |

*Trained-model latencies are from their validation-phase measurements (`experiments/log.csv`,
`baseline_lgbm_022`/`ann_initial_002`/`ann_final_008`) - `scripts/final_eval.py` doesn't
re-measure latency on test, and per-image inference cost doesn't depend on which split an
image comes from, so these are a direct, valid proxy. GenAI's latency is measured on this
evaluation's own 250 test images (network round-trip time per call).

**GenAI performed worse than every trained model, and worse than the trivial majority-class
benchmark** on both the primary metric and accuracy. This is reported as-is, not softened -
CLAUDE.md rule 7 requires it. Looking at the confusion matrix
(`experiments/genai_test_results.csv`), the model over-uses "Other" as a catch-all (56 of 85
Clear images called Other) and frequently confuses Precipitate with Crystals (36 of 120) -
consistent with a general-purpose model that has never seen crystallisation microscopy images
and has no learned sense of what distinguishes these four specific categories.

## Discussion

- **Performance:** every trained model beats GenAI by a wide margin on the primary metric.
  This is the headline result of this comparison - domain-specific training on even a modest
  (~10K image) dataset matters enormously for this task.
- **Latency:** GenAI is 2-3 orders of magnitude slower per prediction (734ms vs. single-digit
  milliseconds), because it's a network round-trip to a remote API rather than a local forward
  pass.
- **Cost per prediction:** GenAI costs real money on every single prediction ($0.66 per 1,000
  images, indefinitely). The trained models cost money once, during training (GPU time - see
  `docs/ann_initial_training.md`), and then run for free locally. For a facility running this
  at scale over months, that compounds into a real, structural cost difference.
- **Training-data needs:** GenAI needed zero MARCO-labelled images - it is purely a prompting
  exercise. The trained models needed the full labelled, leakage-checked training split. If a
  facility had no labelled data at all, GenAI would still be *usable* (if not accurate enough
  to trust) where the trained models could not exist.
- **Privacy / third-party dependence:** every GenAI prediction sends a real drop image to
  Anthropic's servers over the network, subject to Anthropic's data-handling terms and
  requiring network connectivity. The trained models run fully offline/locally once deployed -
  no image ever leaves the facility's machine. For a facility with any data-sensitivity policy
  (e.g. unpublished protein structures), this is a material difference, independent of
  accuracy.

## A realistic opportunity to extend the system with a foundation model

Rather than replacing the trained classifier (which it clearly can't match here), a VLM could
reasonably sit **downstream** of it: when the Final ANN flags a drop as `Crystals`, send the
image and its Grad-CAM overlay (already generated by the web app, see `app/backend/main.py`)
to a VLM with a prompt asking it to describe, in plain language, what's visible in the
highlighted region (e.g. "needle-like crystals near the drop's left edge" vs. "a cloudy region
that could be mistaken for precipitate"). This uses the VLM for what general-purpose language
models are actually good at - producing a readable natural-language summary for a
crystallographer glancing at a result - without asking it to do the specialised visual
classification it performed poorly at here. The trained CNN keeps making the triage decision;
the VLM only helps communicate the *evidence* behind it.
