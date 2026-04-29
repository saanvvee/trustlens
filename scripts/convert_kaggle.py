"""Convert Kaggle outputs (input/prediction/label schema) into the
schema src.evaluate expects, and compute the comparison table directly.

The Kaggle notebook used a binary classification framing (Answer: 0/1)
rather than the structured-JSON framing of src.label_generator.
We extract the predicted class with a regex and map:
    pred=1 -> action="avoid", trust_score=10
    pred=0 -> action="safe",  trust_score=90
If no answer pattern is found we count it as a JSON-validity failure.

Writes:
- data/labeled/baseline_<name>.jsonl
- data/labeled/finetuned.jsonl
- results/comparison_table.md
- results/comparison.png
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score

LABELED = Path("data/labeled")
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

# input: kaggle filename stem -> our destination filename stem
KAGGLE_TO_OURS = {
    "baseline_distilbert.kaggle": "baseline_distilbert",
    "baseline_phi3_zeroshot.kaggle": "baseline_phi3_zeroshot",
    "baseline_phi3_fewshot.kaggle": "baseline_phi3_fewshot",
    "finetuned.kaggle": "finetuned",
}

ANSWER_RE = re.compile(r"Answer:\s*(\d)", re.IGNORECASE)


def extract_pred(row):
    """Return predicted class 0/1 from a Kaggle row, or None if unparseable."""
    pred = row.get("prediction")
    if isinstance(pred, int):
        return pred
    if isinstance(pred, str):
        m = ANSWER_RE.search(pred)
        if m:
            return int(m.group(1))
        # last resort: trailing "0"/"1"
        for tok in pred.strip().split():
            if tok in ("0", "1"):
                return int(tok)
    return None


def convert(src_stem, dst_stem):
    rows_in = [json.loads(l) for l in open(LABELED / f"{src_stem}.jsonl")]
    out = []
    for i, r in enumerate(rows_in):
        pred = extract_pred(r)
        true = int(r.get("label", 0))
        record = {"job_id": i + 1, "fraudulent": true}
        if pred is None:
            # leave action/trust_score absent so it counts as JSON-validity failure
            pass
        else:
            record.update({
                "trust_score": 10 if pred == 1 else 90,
                "red_flags": [],
                "risk_breakdown": {},
                "action": "avoid" if pred == 1 else "safe",
                "reasoning": f"Kaggle predicted class {pred} (binary).",
            })
        out.append(record)

    out_path = LABELED / f"{dst_stem}.jsonl"
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"  {dst_stem}.jsonl: {len(out)} rows ({sum(1 for r in out if 'action' in r)} parseable)")
    return out


REQUIRED = {"trust_score", "action"}


def evaluate(name, rows):
    valid = [r for r in rows if REQUIRED.issubset(r.keys())]
    json_validity = len(valid) / len(rows) if rows else 0.0
    if not valid:
        return {"name": name, "n": len(rows), "json_validity": 0.0,
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "mae": 100.0, "pearson": 0.0, "rouge_l": 0.0}

    y_true = [int(r["fraudulent"]) for r in valid]
    y_pred = [1 if r["action"] == "avoid" else 0 for r in valid]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    truth_score = [10 if t == 1 else 90 for t in y_true]
    pred_score = [int(r.get("trust_score", 50)) for r in valid]
    mae = float(np.mean(np.abs(np.array(truth_score) - np.array(pred_score))))
    pearson = float(np.corrcoef(truth_score, pred_score)[0, 1]) if len(set(pred_score)) > 1 else 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    refs = ["This posting shows multiple red flags consistent with fraud."
            if t == 1 else
            "This posting appears legitimate with standard hiring practices."
            for t in y_true]
    cands = [r.get("reasoning", "") or "" for r in valid]
    rouge_l = float(np.mean([scorer.score(r, c)["rougeL"].fmeasure
                             for r, c in zip(refs, cands)]))

    return {"name": name, "n": len(rows), "json_validity": json_validity,
            "f1": f1, "precision": prec, "recall": rec,
            "mae": mae, "pearson": pearson, "rouge_l": rouge_l}


def main():
    print("converting kaggle outputs...")
    all_results = []
    for src_stem, dst_stem in KAGGLE_TO_OURS.items():
        if not (LABELED / f"{src_stem}.jsonl").exists():
            continue
        rows = convert(src_stem, dst_stem)
        all_results.append(evaluate(dst_stem, rows))

    # write table
    lines = ["# TrustLens — comparison (Kaggle 20-row val subset)\n",
             "| Model | F1 | Precision | Recall | MAE | Pearson r | ROUGE-L | JSON valid | N |",
             "|-------|-----|-----------|--------|-----|-----------|---------|------------|---|"]
    for r in sorted(all_results, key=lambda x: -x["f1"]):
        lines.append(
            f"| {r['name']} | {r['f1']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['mae']:.1f} | {r['pearson']:.2f} | "
            f"{r['rouge_l']:.2f} | {r['json_validity']:.0%} | {r['n']} |")
    (RESULTS / "comparison_table.md").write_text("\n".join(lines))
    print(f"\nwrote {RESULTS / 'comparison_table.md'}")

    # write chart
    names = [r["name"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    plt.figure(figsize=(8, 4))
    plt.bar(names, f1s, color="steelblue")
    plt.ylabel("F1 (action: avoid vs not)")
    plt.title("TrustLens — Kaggle 20-row val comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "comparison.png", dpi=120)
    plt.close()
    print(f"wrote {RESULTS / 'comparison.png'}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
