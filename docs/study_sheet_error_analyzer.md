# Study sheet — `src/error_analyzer.py`

A CLI helper for the rubric's qualitative error analysis. Given a
predictions JSONL, it walks each row, compares the model's `action`
against the ground truth (`fraudulent` 0/1), and writes a CSV of
mismatches to `results/errors_for_review.csv` for human review.

## What `categorise(row, true_fraud)` decides

It returns one of these strings, or `None` if the row is fine:

| Category | When it fires |
|---|---|
| `format_break` | The row is missing one of the required keys (`trust_score`, `red_flags`, `action`, `reasoning`) — the model couldn't even produce the right shape. |
| `missed_red_flag` | True fraud, but the model said `safe` *and* emitted no red_flags. Worst kind of error. |
| `wrong_action` | True fraud, model said `safe`/`caution`, but did emit some red_flags — the signals were there, the verdict didn't follow. |
| `false_red_flag` | True real, model said `avoid` — annoying for users; flags a legit role as a scam. |
| `score_miscalibrated` | Action is right but `trust_score` lands on the wrong side of 50 (e.g., action=avoid but score=80). |

`hallucinated_company` is in the documented category list but isn't
auto-detected — it requires reading the `reasoning` text and
checking whether named companies actually exist. Reviewers can
re-label any row to `hallucinated_company` after reading.

## Inputs and outputs

**Input:** any predictions JSONL with `job_id`, `fraudulent`,
`action`, `trust_score`, `red_flags`, `reasoning`. That includes
every baseline JSONL and the fine-tuned model's predictions.

**Output:** `results/errors_for_review.csv` with columns:
`job_id`, `true_fraud`, `predicted_action`, `trust_score`,
`suggested_category`, `reasoning` (truncated to 200 chars),
`job_text_preview` (looked up from val.csv by job_id, truncated).

The reviewer opens this CSV in any spreadsheet tool, fixes the
`suggested_category` column where the heuristic was wrong, and uses
counts of the corrected categories in the report.

## Why "suggested" not "final" categorisation

The categories `wrong_action`, `missed_red_flag`, `false_red_flag`,
`score_miscalibrated`, `format_break` can all be detected
deterministically from the data. `hallucinated_company` cannot —
the only way to know is to read what the model wrote. Rather than
auto-flagging every row containing a company name, we **suggest**
the simplest defensible category and let the reviewer escalate.
This keeps the script honest: it never claims to have detected an
error class it can't.

## Sample viva Q&A

**Q: Why is `hallucinated_company` left for manual review?**
A: Detecting it requires entity-linking the company names in
`reasoning` against a ground-truth corporate registry — that's a
project on its own. The simpler categories give us 80% of the
signal automatically; for the last 20% we read the rows ourselves.
For a course project that's the right place to draw the
automation line.

**Q: How does the suggested category change my F1 number?**
A: It doesn't. F1 / MAE / ROUGE all live in `evaluate.py` and use
binary or numeric comparisons, not category labels. The
error-analysis CSV is a *qualitative* artefact for the report,
separate from the *quantitative* metrics. The rubric explicitly
asks for both.

**Q: What's an example of a row this would flag as
`score_miscalibrated`?**
A: A model that says `action="avoid"` but emits `trust_score=72`.
The two outputs disagree internally — either the model thinks the
posting is dangerous (action=avoid) or it doesn't (score=72). This
is most often a sign of weak instruction-following on the JSON
schema, and it's a useful flag because it surfaces a specific
training-data issue we can fix in a future iteration.
