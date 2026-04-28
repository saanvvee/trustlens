"""CLI for error analysis on a predictions JSONL.

Walks each row, compares its predicted ``action`` against the ground
truth (``fraudulent`` 0/1), and writes a CSV of mismatches with a
suggested error category for human review.

Categories (the human reviewer can override any):
    hallucinated_company  — model claimed a company is verified that isn't
    format_break          — output JSON is missing required fields
    score_miscalibrated   — trust_score wildly disagrees with action
    wrong_action          — action conflicts with the ground-truth label
    missed_red_flag       — fraud=1 but model emitted no red_flags
    false_red_flag        — fraud=0 but model flagged it as a scam

Usage:
    python -m src.error_analyzer \\
        --predictions data/labeled/baseline_distilbert.jsonl \\
        --output results/errors_for_review.csv
"""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

REQUIRED = {"trust_score", "red_flags", "action", "reasoning"}


def categorise(row, true_fraud):
    """Return the most likely error label, or None if the row is fine."""
    if not REQUIRED.issubset(row.keys()):
        return "format_break"

    action = row.get("action", "")
    score = int(row.get("trust_score", 50))
    flags = row.get("red_flags", []) or []
    pred_fraud = 1 if action == "avoid" else 0

    if true_fraud == 1 and pred_fraud == 0:
        return "missed_red_flag" if not flags else "wrong_action"
    if true_fraud == 0 and pred_fraud == 1:
        return "false_red_flag"

    # action correct but score is on the wrong side
    if pred_fraud == 1 and score > 50:
        return "score_miscalibrated"
    if pred_fraud == 0 and score < 50:
        return "score_miscalibrated"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="JSONL file with job_id, fraudulent, action, etc.")
    p.add_argument("--val", default="data/processed/val.csv",
                   help="for looking up job_text by job_id")
    p.add_argument("--output", default="results/errors_for_review.csv")
    args = p.parse_args()

    val = pd.read_csv(args.val).set_index("job_id")

    rows = [json.loads(l) for l in open(args.predictions) if l.strip()]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_err = 0
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "true_fraud", "predicted_action", "trust_score",
                    "suggested_category", "reasoning", "job_text_preview"])
        for r in rows:
            jid = int(r.get("job_id", -1))
            true_fraud = int(r.get("fraudulent", val.loc[jid, "fraudulent"]
                                    if jid in val.index else 0))
            cat = categorise(r, true_fraud)
            if cat is None:
                continue
            n_err += 1
            jt = val.loc[jid, "job_text"] if jid in val.index else ""
            w.writerow([
                jid, true_fraud,
                r.get("action", ""), r.get("trust_score", ""),
                cat,
                (r.get("reasoning", "") or "")[:200],
                str(jt)[:200],
            ])
    print(f"wrote {n_err} error rows to {args.output} (out of {len(rows)} predictions)")


if __name__ == "__main__":
    main()
