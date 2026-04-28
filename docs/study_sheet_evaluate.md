# Study sheet — `src/evaluate.py`

Reads every `data/labeled/baseline_*.jsonl` (and the fine-tuned
model's predictions when present), computes per-model metrics,
writes `results/comparison_table.md` + `results/comparison.png`, and
logs each row to SQLite `eval_runs`.

## What gets measured

For each predictions file:

| Metric | What it tells us | Computed from |
|---|---|---|
| **F1** on action | Did the model correctly call out fraud (action="avoid") vs everything else? | sklearn.metrics.f1_score on y_pred=avoid?1:0, y_true=fraudulent |
| **Precision** | Of the things flagged as scam, how many really were? | sklearn.metrics.precision_score |
| **Recall** | Of the real scams, how many did we catch? | sklearn.metrics.recall_score |
| **MAE** on trust_score | Average distance between predicted score and a synthetic ground truth (90 if real, 10 if fraud) | numpy mean abs |
| **Pearson r** on trust_score | Correlation between predicted scores and the synthetic truth | numpy corrcoef |
| **ROUGE-L** on reasoning | Lexical overlap between the model's reasoning and a generic per-class reference | rouge_score lib |
| **JSON-validity** | Fraction of rows that parsed and have all required keys | len(valid)/len(rows) |

## The synthetic ground-truth trick

Val.csv only has the `fraudulent` 0/1 label — no per-row reasoning
or trust_score from a human. To compute MAE / ROUGE-L we synthesise
trivial gold answers: trust_score = 90 if fraudulent==0 else 10;
reasoning = "this posting appears legitimate ..." or "this posting
shows multiple red flags ...". This is a **directional** measure,
not an absolute one — useful for *comparing baselines against each
other*, less useful as an absolute score.

For absolute reasoning quality you'd label val with a strong teacher
(STEP 8 currently labels train.jsonl only) — that's a clear future-
work item we name in the README.

## Action mapping

The model's `action` is 3-class (avoid / caution / safe) but
`fraudulent` is 2-class. We collapse:
- `avoid` → predicted_fraud = 1
- `caution` or `safe` → predicted_fraud = 0

This is conservative — we treat "caution" as "did not flag as
fraud." Rationale: a real fraud detection system that says
"caution" rather than "avoid" has effectively let the user fall for
it. The user wanted a strong opinion.

## Outputs

- **`results/comparison_table.md`** — markdown table sorted by F1,
  one row per model, ready to drop into the README.
- **`results/comparison.png`** — bar chart of F1 across models for
  the report figure.
- **`eval_runs` table** — one SQLite row per model per pass,
  timestamped, so we can show "we ran this evaluation N times" in
  viva.

## Sample viva Q&A

**Q: Why F1 on action and not accuracy?**
A: With ~5% fraud, accuracy is misleading — a model that always
predicts "safe" gets 95% accuracy and catches zero scams. F1 weighs
precision and recall equally, so it punishes the always-safe
strategy.

**Q: Why ROUGE-L specifically?**
A: ROUGE-L measures longest-common-subsequence overlap, which is
robust to small reorderings — the same idea expressed in slightly
different words still scores high. ROUGE-1 (unigram overlap) is too
forgiving; ROUGE-2 is too strict for short reasoning paragraphs.
ROUGE-L is the sweet spot for evaluating short LLM-generated
explanations against a reference.

**Q: How do you handle predictions where the JSON parser failed?**
A: They count toward `json_validity` (denominator includes them,
numerator doesn't) but are excluded from F1 / MAE / ROUGE
calculations — there's no `action` to compare. This means a model
that produces clean JSON 100% of the time but with worse content
will get a *lower* F1 than a model that's more often correct on the
rows it did parse. That trade-off is exactly what we want to
expose: JSON discipline is a separate axis from content quality.
