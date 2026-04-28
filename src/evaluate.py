"""Evaluation suite for trust-assessment predictions.

Loads every ``data/labeled/baseline_*.jsonl`` (and
``data/labeled/finetuned.jsonl`` if it exists), computes per-model
metrics, writes a markdown comparison table and a bar chart, and
logs each row to SQLite ``eval_runs``.

Metrics:
- F1, precision, recall on action  (avoid -> 1, else 0; vs ground truth fraudulent)
- MAE on trust_score             (vs synthetic truth: 90 if real, 10 if fraud)
- Pearson r on trust_score
- ROUGE-L on reasoning vs a generic per-class reference
- JSON-validity rate              (fraction of rows with all required keys)

Run from project root:
    python -m src.evaluate
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.db import log_eval_run

VAL_CSV = "data/processed/val.csv"
LABELED_DIR = Path("data/labeled")
RESULTS_DIR = Path("results")
REQUIRED_KEYS = {"trust_score", "red_flags", "risk_breakdown", "action", "reasoning"}


def _load_val_truth():
    df = pd.read_csv(VAL_CSV)
    return df.set_index("job_id")["fraudulent"].astype(int).to_dict()


def _load_predictions(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _evaluate(name, path, truth):
    rows = _load_predictions(path)
    if not rows:
        return None

    valid = [r for r in rows if REQUIRED_KEYS.issubset(r.keys())
             and int(r.get("job_id", -1)) in truth]
    json_validity = len(valid) / len(rows) if rows else 0.0

    if not valid:
        return {"name": name, "n": len(rows), "json_validity": json_validity,
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "mae": 100.0, "pearson": 0.0, "rouge_l": 0.0}

    y_true = [truth[int(r["job_id"])] for r in valid]
    y_pred = [1 if r["action"] == "avoid" else 0 for r in valid]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    truth_score = [10 if t == 1 else 90 for t in y_true]
    pred_score = [int(r.get("trust_score", 50)) for r in valid]
    mae = float(np.mean(np.abs(np.array(truth_score) - np.array(pred_score))))
    pearson = float(np.corrcoef(truth_score, pred_score)[0, 1]) if len(truth_score) > 1 else 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    references = [
        "This posting shows multiple red flags consistent with fraud."
        if t == 1 else
        "This posting appears legitimate with standard hiring practices."
        for t in y_true
    ]
    candidates = [r.get("reasoning", "") or "" for r in valid]
    rouge_l = float(np.mean([scorer.score(ref, cand)["rougeL"].fmeasure
                             for ref, cand in zip(references, candidates)]))

    return {"name": name, "n": len(rows), "json_validity": json_validity,
            "f1": f1, "precision": prec, "recall": rec,
            "mae": mae, "pearson": pearson, "rouge_l": rouge_l}


def _write_table(results):
    lines = ["# TrustLens — baseline comparison\n",
             "| Model | F1 | Precision | Recall | MAE | Pearson r | ROUGE-L | JSON valid | N |",
             "|-------|-----|-----------|--------|-----|-----------|---------|------------|---|"]
    for r in sorted(results, key=lambda x: -x["f1"]):
        lines.append(
            f"| {r['name']} | {r['f1']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['mae']:.1f} | {r['pearson']:.2f} | "
            f"{r['rouge_l']:.2f} | {r['json_validity']:.0%} | {r['n']} |")
    (RESULTS_DIR / "comparison_table.md").write_text("\n".join(lines))
    print(f"wrote {RESULTS_DIR / 'comparison_table.md'}")


def _write_chart(results):
    names = [r["name"] for r in results]
    f1s = [r["f1"] for r in results]
    plt.figure(figsize=(8, 4))
    plt.bar(names, f1s, color="steelblue")
    plt.ylabel("F1 score (action: avoid vs not)")
    plt.title("TrustLens — baseline comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison.png", dpi=120)
    plt.close()
    print(f"wrote {RESULTS_DIR / 'comparison.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    truth = _load_val_truth()

    paths = list(LABELED_DIR.glob("baseline_*.jsonl"))
    if (LABELED_DIR / "finetuned.jsonl").exists():
        paths.append(LABELED_DIR / "finetuned.jsonl")
    if not paths:
        print(f"no predictions in {LABELED_DIR} — run src.baselines first")
        return

    results = []
    for p in sorted(paths):
        name = p.stem.replace("baseline_", "")
        r = _evaluate(name, p, truth)
        if not r:
            continue
        results.append(r)
        log_eval_run(name, r["rouge_l"], r["f1"], r["mae"], r["json_validity"])
        print(f"{name:20s}  F1={r['f1']:.3f}  MAE={r['mae']:5.1f}  "
              f"valid={r['json_validity']:.0%}  n={r['n']}")

    if results:
        _write_table(results)
        _write_chart(results)


if __name__ == "__main__":
    main()
