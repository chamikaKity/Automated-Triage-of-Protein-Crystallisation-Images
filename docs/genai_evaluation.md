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

**Stage 1 (used to pick the official prompt).** Two prompt variants were compared on 12
**validation** images (3 per class, seed=42), never test:

| Prompt | Validation accuracy |
|---|---|
| `crystals_classification_v1.txt` | 6/12 |
| `crystals_classification_v2.txt` (added more explicit visual cues for Precipitate/Other/Crystals) | 4/12 |

v2's extra guidance made the model over-call "Crystals" on images that weren't (several
Precipitate validation images flipped to confident Crystals predictions). **v1 was selected**
and locked in before any test image was touched, and is what the official test-set result below
uses.

**Stage 2 (a properly-powered follow-up, also validation-only).** 12 images is a very small
comparison - noisy enough that it's worth distrusting. With time still available,
`scripts/eval_genai.py --split val --n-sample 250` was added (the existing script only read
test; `--split` now lets prompt experiments stay on validation, never touching test) and all
four prompt variants tried so far were compared on the same stratified 249-image validation
sample:

| Prompt | What it changes | Accuracy | Crystals PR-AUC | recall@P_MIN |
|---|---|---|---|---|
| v1 (official) | baseline descriptive prompt | 0.273 | 0.149 | 0.000 |
| v2 | extra visual cues per class | 0.269 | 0.113 | 0.000 |
| v3 | minimal - just the four class names, no descriptive cues | 0.394 | 0.140 | 0.032 |
| v4 | conservative - requires 2+ clear facets before calling "Crystals" | **0.386** | **0.235** | **0.194** |

Two new variants were written with explicit, pre-stated hypotheses before running them: v3 was
expected to underperform v1 (hypothesis: stripping domain-specific visual guidance should hurt
a general-purpose model on this niche task), and v4 was expected to underperform by being too
conservative and suppressing Crystals recall. **Both hypotheses turned out to be wrong** - v3
and v4 both clearly beat v1 on every metric except v3's marginally lower PR-AUC, with v4 the
best prompt of any tried. This is reported as-is, not reframed as a win after the fact: it
means the original 12-image comparison in Stage 1 was underpowered and likely did not identify
the best available prompt - a genuine limitation of that earlier methodology.

**v2 is logged and used as one of the coursework's required ≥4 unsuccessful experiments**
(`genai_prompt_v2_val_000` in `experiments/log.csv`) - it underperforms v1 on both accuracy and
PR-AUC, confirmed now at n=249 rather than the original n=12. v3 and v4 do not count as
failures, since they succeeded; they're kept here as an honest secondary finding instead.

**The official test-set result below was not changed.** Per the test-isolation rule, prompt
choice is locked in before test is touched, and this follow-up finding - however tempting -
does not justify re-running the test evaluation with v4. All four prompt files are kept in
`experiments/prompts/` as the full record.

## Test-set evaluation

Evaluated on the **full 2,053-image test set**. A smaller stratified sample of 250 was run
first (results kept as `genai_claude_haiku_000` in `experiments/log.csv` and
`experiments/genai_test_results.csv`, for the record) to get a real cost/latency measurement
before committing to the full run - it cost $0.16 and took a few minutes, which turned out
cheap enough that there was no real reason not to run the full set, so that was done too
(`genai_claude_haiku_full_000`, `experiments/genai_test_results_full.csv`). The two runs agree
closely (accuracy 0.264 vs. 0.262, recall@P_MIN 0.000 in both), so the smaller sample's
headline conclusion held up - the full run below is the one used for the official comparison
table, being the more statistically robust and directly comparable to every other model's
full-test-set evaluation.

## Comparison with every other model (test set)

| Model | recall@P_MIN | Crystals PR-AUC | Accuracy | Inference latency* | One-time training cost | Marginal cost / 1000 predictions | Training data needed |
|---|---|---|---|---|---|---|---|
| Trivial benchmark | 0.000 | 0.124 | 0.480 | ~0 ms | ~$0 (local CPU, <1s) | $0 | none |
| Classical baseline (LightGBM) | 0.937 | 0.678 | 0.812 | 0.03 ms (CPU) | ~$0 (local CPU, 29s) | $0 | ~10,261 labelled images |
| Initial ANN (frozen) | 0.886 | 0.672 | 0.715 | 6.3 ms (GPU) | ~$1.21 (RTX 4090, 3.57 GPU-hrs) | $0 | ~10,261 labelled images |
| Final ANN (layer4) | **0.965** | **0.823** | 0.809 | 6.2 ms (GPU) | ~$1.22 (RTX 4090, 3.59 GPU-hrs) | $0 | ~10,261 labelled images |
| **GenAI (Claude Haiku 4.5), n=2,053** | **0.000** | 0.147 | 0.262 | 746 ms | $0 (no training step) | **$0.66, forever** | **zero** |

*Trained-model latencies are from their validation-phase measurements (`experiments/log.csv`,
`baseline_lgbm_022`/`ann_initial_002`/`ann_final_008`) - `scripts/final_eval.py` doesn't
re-measure latency on test, and per-image inference cost doesn't depend on which split an
image comes from, so these are a direct, valid proxy. GenAI's latency is measured on this
evaluation's own test images (network round-trip time per call). Total GenAI cost for all
2,053 images: $1.35.

**"Training data needed = zero" for GenAI and "marginal cost = $0" for the trained models are
two different claims, not the same thing** - worth being precise about this in the report. The
former is about labelled images (GenAI needs none, zero-shot); the latter is about the cost of
a prediction *after* training is already done (a local forward pass, no further spend). Neither
means ANN training itself was free: it cost roughly $2.43 in real GPU rental (Initial + Final
ANN searches combined, RunPod RTX 4090, from the logged `train_time_s` in `experiments/log.csv`
converted at the actual hourly rate paid), plus an estimated further $3-5 lost to unreliable
rented GPU pods during this project (stuck/broken instances killed before producing results,
not attributable to any one model) - call it **~$6-8 total GPU spend** for the ANN training
side of this project. GenAI's cost shape is structurally different: no training cost at all,
but an unbounded *per-prediction* cost that would eventually overtake the ANNs' one-time
training cost at any real deployment scale.

**GenAI performed worse than every trained model, and worse than the trivial majority-class
benchmark** on both the primary metric and accuracy. This is reported as-is, not softened -
CLAUDE.md rule 7 requires it. Looking at the full confusion matrix
(`experiments/genai_test_results_full.csv`, order Clear/Crystals/Other/Precipitate):

```
            pred: Clear  Crystals  Other  Precipitate
true Clear       223       44      418      16
true Crystals      13      79      102      60
true Other          16      22       45      30
true Precipitate   107     365      323     190
```

The model over-uses "Other" as a catch-all (418 of 701 Clear images called Other - the single
largest error in the whole matrix) and frequently confuses Precipitate with Crystals (365 of
985) - consistent with a general-purpose model that has never seen crystallisation microscopy
images and has no learned sense of what distinguishes these four specific categories. Notably,
Crystals recall in raw terms isn't terrible (79/254 = 31% caught, plus partial credit via
likelihood scores) but precision collapses because Precipitate is so often also called
Crystals - exactly the kind of false-alarm-heavy behaviour the primary metric (recall at a
precision floor) is designed to penalise.

## Discussion

- **Performance:** every trained model beats GenAI by a wide margin on the primary metric.
  This is the headline result of this comparison - domain-specific training on even a modest
  (~10K image) dataset matters enormously for this task.
- **Latency:** GenAI is 2-3 orders of magnitude slower per prediction (746ms vs. single-digit
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
