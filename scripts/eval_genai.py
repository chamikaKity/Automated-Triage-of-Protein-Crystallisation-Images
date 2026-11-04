"""Evaluate a pretrained vision-language model (Claude) on a sample of the TEST split.

Reads test images only to score the GenAI baseline - the prompt itself (experiments/prompts/
crystals_classification_v1.txt) was selected using validation images only (see the comparison
against v2, run separately and documented in docs/genai_evaluation.md), per CLAUDE.md's rule
that prompt choices use train/validation only.

Evaluates a stratified random sample (not the full ~2,053 test images) for cost and time -
documented explicitly, not hidden. Logs cost, latency, and the primary metric alongside every
other model in experiments/log.csv.
"""

import argparse
import base64
import io
import json
import re
import time
from datetime import date
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix

CLASS_ORDER = ["Clear", "Crystals", "Other", "Precipitate"]
MODEL = "claude-haiku-4-5-20251001"
SEED = 42
N_SAMPLE = 250
# Haiku 4.5 pricing at time of writing, per million tokens - see docs/genai_evaluation.md.
PRICE_PER_MTOK_INPUT = 1.00
PRICE_PER_MTOK_OUTPUT = 5.00


def recall_at_precision(y_true_binary: np.ndarray, scores: np.ndarray, p_min: float = 0.30) -> float:
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true_binary, scores)
    achievable = recall[precision >= p_min]
    return float(achievable.max()) if len(achievable) else 0.0


def parse_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model response: {text!r}")
    return json.loads(match.group(0))


def classify(client: anthropic.Anthropic, prompt: str, image_path: Path) -> tuple[dict, int, int, float]:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    t0 = time.time()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    latency_s = time.time() - t0
    result = parse_response(resp.content[0].text)
    return result, resp.usage.input_tokens, resp.usage.output_tokens, latency_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, default=Path("data/splits.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--prompt-path", type=Path,
                         default=Path("experiments/prompts/crystals_classification_v1.txt"))
    parser.add_argument("--log-path", type=Path, default=Path("experiments/log.csv"))
    parser.add_argument("--results-out", type=Path, default=Path("experiments/genai_test_results.csv"))
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--id", type=str, default="genai_claude_haiku_000",
                         help="Log row id - override to avoid colliding with a previous run's row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = anthropic.Anthropic()
    prompt = args.prompt_path.read_text()

    splits = pd.read_csv(args.splits_path)
    test_df = splits[splits["split"] == "test"]
    # Stratified sample: roughly proportional to each class's share of the full test set.
    frac = args.n_sample / len(test_df)
    sample = test_df.groupby("label", group_keys=False).sample(
        frac=frac, random_state=args.seed
    ).reset_index(drop=True)
    print(f"test set: {len(test_df)} images total, sampled {len(sample)} (read ONLY here, per hard rule 2)")
    print(sample["label"].value_counts())

    rows = []
    total_in_tok, total_out_tok, total_latency = 0, 0, 0.0
    for i, row in sample.iterrows():
        try:
            result, in_tok, out_tok, latency_s = classify(client, prompt, args.data_dir / row["path"])
        except Exception as e:
            print(f"  [{i + 1}/{len(sample)}] FAILED on {row['path']}: {e}")
            continue
        total_in_tok += in_tok
        total_out_tok += out_tok
        total_latency += latency_s
        rows.append({
            "path": row["path"], "true_label": row["label"],
            "predicted_class": result["predicted_class"],
            "crystals_likelihood": result["crystals_likelihood"],
        })
        if (i + 1) % 25 == 0:
            print(f"  [{len(rows)}/{len(sample)}] done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.results_out, index=False)

    y_true = results_df["true_label"].to_numpy()
    preds = results_df["predicted_class"].to_numpy()
    y_bin = (y_true == "Crystals").astype(int)
    crystals_scores = results_df["crystals_likelihood"].to_numpy() / 100.0

    accuracy = float((preds == y_true).mean())
    crystals_pr_auc = float(average_precision_score(y_bin, crystals_scores))
    recall = recall_at_precision(y_bin, crystals_scores)
    cm = confusion_matrix(y_true, preds, labels=CLASS_ORDER).tolist()

    cost_usd = (total_in_tok / 1e6 * PRICE_PER_MTOK_INPUT) + (total_out_tok / 1e6 * PRICE_PER_MTOK_OUTPUT)
    avg_latency_ms = total_latency / len(rows) * 1000

    print(f"\nn={len(rows)} accuracy={accuracy:.3f} crystals_pr_auc={crystals_pr_auc:.3f} recall@P_MIN={recall:.3f}")
    print(f"cost: ${cost_usd:.4f} total (${cost_usd / len(rows) * 1000:.4f} per 1000 images)")
    print(f"avg latency: {avg_latency_ms:.0f} ms/image")
    print(f"confusion matrix (order={CLASS_ORDER}): {cm}")

    log_row = {
        "id": args.id,
        "date": date.today().isoformat(),
        "model": f"GenAI ({MODEL}) - TEST sample n={len(rows)}",
        "what_changed": f"prompt={args.prompt_path.name}, sample_frac={frac:.3f}",
        "hyperparameters": json.dumps({"model": MODEL, "prompt": args.prompt_path.name}),
        "seed": args.seed,
        "validation_metrics": json.dumps({
            "accuracy": accuracy,
            "crystals_pr_auc": crystals_pr_auc,
            "recall_at_p_min": recall,
            "confusion_matrix": cm,
            "cost_usd_total": round(cost_usd, 4),
            "cost_usd_per_1000_images": round(cost_usd / len(rows) * 1000, 4),
            "inference_latency_ms_per_image": round(avg_latency_ms, 1),
            "n_images": len(rows),
        }),
        "notes": "TEST SET sample result - scripts/eval_genai.py",
    }
    pd.DataFrame([log_row]).to_csv(args.log_path, mode="a", header=False, index=False)
    print(f"\nlogged to {args.log_path}, raw results saved to {args.results_out}")


if __name__ == "__main__":
    main()
