# CLAUDE.md – MARCO Crystallisation Image Triage (AI Coursework)

Individual MSc Data Science AI coursework (Coventry / NIBM), 100% of the module mark.
Approved by the lecturer. Deadline: **6 November 2026**.
Full brief: `docs/brief.pdf`. Approved proposal: `docs/proposal.pdf`.
Changing the problem or dataset requires lecturer approval – do not suggest it casually.

## Project summary
- **User:** a crystallographer or crystallisation-facility scientist reviewing imaged plates.
- **Decision:** prioritise drops flagged as likely to contain crystals for inspection and harvesting
  for beamline data collection. The system supports the decision; it never discards drops automatically.
- **Framing:** a synchrotron-facility-style MX workflow, at the crystallisation-screening stage before
  beamline data collection. Do not claim affiliation with or endorsement by any facility.
- **Data:** MARCO (https://marco.ccr.buffalo.edu), CC BY 4.0, JPEG version, extracted from TFRecords.
  The full set is over 500 MB: submit a download/extract script plus a representative sample.
- **Task:** 4-class image classification – `Clear`, `Crystals`, `Precipitate`, `Other`.
  `Crystals` is the decision-relevant minority class.
- **Costs:** a false negative (missed crystal) can lose a hit from scarce protein; a false positive
  costs a few minutes of inspection. The loss, metric and threshold must reflect this asymmetry.

## Metrics
- **Primary:** recall on `Crystals` at a fixed minimum precision (`P_MIN = TBD`, set from EDA,
  using validation data only). The same primary metric is used for every model, including GenAI.
- **Supporting:** PR-AUC for `Crystals`, per-class confusion matrix, training time, inference latency.
- Do not use overall accuracy as the headline metric.

## Hard rules (never break these)
1. **One fixed split.** Always load `data/splits.csv`. Never re-split, re-shuffle or resample across
   splits. The baseline, every ANN variant and the GenAI evaluation all use this split.
2. **Test isolation.** The test split is read **only** by `scripts/final_eval.py`, run once at the end.
   Model selection, thresholds, hyperparameters and prompt choices use train/validation only.
   If code anywhere else touches test data, stop and flag it.
3. **Leakage.** Images of the same drop (same plate/well, or a near-duplicate group) must stay in one
   split. Do not add features that would not exist at prediction time. Document every leakage check.
4. **Seeds.** Fix seeds for Python, NumPy and PyTorch, and record them. Where results vary materially
   by seed, report the mean and spread over at least 3 seeds.
5. **Experiment log.** Every substantive train/eval run appends one row to `experiments/log.csv`:
   id, date, model, what changed, hyperparameters, seed, validation metrics, notes / what was learned.
   Never edit or delete past rows. Failed runs are logged too.
6. **Fair comparison.** The classical baseline gets a tuning budget of the same order of magnitude as
   the ANN. Record the search space and approximate number of configurations for each model family.
7. **Honest results.** Never fabricate, round favourably, hand-edit results, weaken the baseline or
   omit failed experiments. If the baseline beats the ANN, report it unchanged and explain it with
   evidence. Every number in the report must trace back to the log and be reproducible from code.
8. **Pre-registered expectation (Q2.b).** Before running `final_eval.py`, write which model family
   (ANN or classical) is expected to win on the primary metric, and why, in
   `docs/expectation.md` (dated, committed). Never edit it afterwards.
9. **Originality.** Do not copy or adapt code from existing MARCO implementations or other public
   repositories (lecturer requirement). If an idea comes from published work, cite it in
   `docs/references.md`.
10. **Request-time predictions.** The web app must run the trained model on each request. Never
    hard-code or precompute predictions.

## Coursework minimums (confirmed by lecturer where noted)
- At least **4** unsuccessful experiments, logged and diagnosed (lecturer confirmed).
- At least **3** incorrect predictions analysed with XAI in the report; at least **1** shown in the
  video (lecturer confirmed).
- At least **3** hyperparameters from learning rate, batch size, network depth (others need lecturer
  approval), with conclusions supported by results. These three are the required set.
- **Optional 4th – degree of fine-tuning** (frozen backbone / partial unfreeze / full fine-tuning).
  **Pending lecturer approval:** discuss with Dr. Rajadurai when we reach ANN training. Do not run
  or report it until approved, and record the approval in `docs/`. It must not replace any of the
  required three.
- At least **2** optimisers compared empirically and related to their mechanics.
- At least **3** XAI approaches compared (architecture-specific and model-agnostic); at least
  **2** sanity-check cases (e.g. occluding the highlighted region and checking the prediction changes).

## Deliverables by question (report follows Q1–Q6 order)
- **Q1 EDA:** dataset description and licence; class and source distributions; leakage checks;
  FP/FN consequences; metric justification; end with **3 findings that constrain modelling**.
- **Q2 Design:** architecture diagram, layers, dimensions, activations, parameter count; justify
  depth, width, normalisation, regularisation, output layer and loss individually; alternatives
  considered and why rejected; pre-registered expectation (rule 8). Prefer fine-tuning a pretrained
  backbone over training from scratch.
- **Q3 Development and optimisation:** learning curves with an under/overfitting diagnosis.
  Comparison table: trivial benchmark, classical baseline, initial ANN, final ANN – each with
  primary metric, training time and inference latency. Error analysis by class, source institution
  and image quality.
- **Q4 Web app:** frontend and backend; the user uploads an image and gets a prediction at request
  time. Show the class score and explain what it means – never call an uncalibrated probability a
  guaranteed confidence. State limitations (e.g. unseen imagers). Test normal, invalid and
  out-of-distribution inputs.
- **Q5 XAI:** Grad-CAM in the app as the primary method, compared with occlusion and LIME/SHAP.
  Present it in a way a crystallographer can use. Never claim an explanation proves causal reasoning
  or the true reason for a prediction; discuss limitations.
- **Q6 GenAI:** a pretrained vision-language model on the **same test images** with the **same
  primary metric**. Prompts selected on validation only and saved in `experiments/prompts/`.
  Record model/version, configuration, services used and costs. Compare performance, latency,
  cost per prediction, training-data needs, privacy and third-party dependence. Give at least one
  realistic opportunity for a foundation model to extend the system.
- **Submission:** code with README and a pinned environment file; experiment log; data sample plus
  download script; report of 25 pages max excluding appendices; video of 8 minutes max showing a
  real-time prediction, the explanation feature and at least 1 incorrect prediction; AI-use
  declaration appendix. A live presentation follows, so all code must be explainable.

## Repository layout
```
data/          raw/ (not committed), sample/, splits.csv
src/marco/     package: data loading, models, training, evaluation, XAI, GenAI evaluation
scripts/       download_extract.py, make_splits.py, train.py, final_eval.py
experiments/   log.csv, configs/, prompts/
app/           web application (backend + frontend)
notebooks/     EDA and analysis; import from src/, don't duplicate logic
tests/         split, leakage, test-isolation and data-pipeline tests
docs/          brief, proposal, expectation, references, ai_use_log
```

## Stack and conventions
- Python, PyTorch, src-layout package with `pyproject.toml` and a pinned environment.
- Lint with `ruff`; test with `pytest`. Keep functions small with type hints and docstrings.
- Commands: `TBD` (fill in once the environment is set up).

## How I want help
- **Explain before writing.** For any non-trivial code, explain the approach first. I must be able
  to explain and modify every line in the live presentation.
- When I ask to learn or try something myself, guide me with hints instead of writing the solution.
- Keep changes small and focused; don't refactor unrelated code.
- Ask before adding dependencies, downloading data or running long training jobs.
- Point out anything that risks breaking the hard rules, even if I didn't ask.
- AI-tool use must be declared in the report appendix: add a one-line note to `docs/ai_use_log.md`
  for any substantive help (date, task, what was produced, what I verified).
