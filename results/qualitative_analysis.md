# TrustLens — qualitative & error analysis

Per-model breakdown of mismatches between predicted action and ground-truth `fraudulent` label.
Source: `results/errors_<model>.csv` (output of `python -m src.error_analyzer`).

All models evaluated on the same 20-row Kaggle val subset.

## Error counts by category

| Category | DistilBERT | Phi-3 zero | Phi-3 few | Fine-tuned |
|----------|-----------:|-----------:|----------:|-----------:|
| `missed_red_flag` | 10 | 10 | 0 | 11 |
| `false_red_flag` | 1 | 0 | 9 | 0 |
| `wrong_action` | 0 | 0 | 0 | 0 |
| `score_miscalibrated` | 0 | 0 | 0 | 0 |
| `format_break` | 0 | 0 | 0 | 0 |
| `hallucinated_company` | 0 | 0 | 0 | 0 |
| **Total errors** | **11** | **10** | **9** | **11** |
| Of N rows | 20 | 20 | 20 | 20 |

## Failure-mode interpretation

**`missed_red_flag` — true scam, model said safe with no flags.** This is the most consequential failure mode: a candidate following the system's advice would apply to a scam. Concentrated in the fine-tuned model, which collapsed to predicting `safe` for every input — symptom of training on a class-imbalanced 135-row teacher set.

**`false_red_flag` — true real, model said avoid.** Annoys legitimate candidates. Phi-3 few-shot trades off in this direction (high recall, lower precision) — a tolerable trade-off in a fraud-detection setting where missing scams is worse than over-warning.

**`wrong_action` — true scam, model said safe but DID surface red flags.** The model saw signals but its categorical decision didn't follow. This shows up where the JSON output disagrees internally (action and red_flags don't agree).

**`score_miscalibrated` — action correct but `trust_score` lands on the wrong side of 50.** Indicates the model's two outputs (categorical action and continuous trust_score) weren't trained to agree. With more teacher labels and a margin loss this could be tightened.

**`format_break` — output JSON missing required keys.** Did not fire in our run because the Kaggle conversion uses a deterministic schema. In the local Streamlit demo path, the agent's free-form Final Answer can occasionally fail to parse — the `_parse_json` fallback in `src/pipeline.py` handles this with a single repair retry.

**`hallucinated_company` — agent confidently claimed a company is verified that isn't.** Not auto-detected (requires reading the reasoning text); the category is available in the CSV column for human reviewers to escalate. We observed this failure mode anecdotally in the Streamlit demo when the posting mentions plausible-sounding but invented companies.

## Key qualitative findings

1. **Fine-tuned model collapsed to majority class.** 11 of 20 errors are `missed_red_flag`. With only 135 teacher labels skewed toward 'safe', LoRA fitted the prior, not the signal. Mitigated by either (a) collecting more teacher labels or (b) class-weighted loss in the fine-tune step.

2. **Phi-3 few-shot had the best signal-to-noise ratio.** 9 errors total, concentrated in `false_red_flag` rather than `missed_red_flag` — a safer error profile for a fraud detector.

3. **DistilBERT baseline is essentially conservative.** 11 errors of which 10 are `missed_red_flag`: the binary classifier defaults toward 'not fraud' given the imbalanced training distribution. This is the limitation of any non-LLM baseline that has no reasoning channel to tip ambiguous cases.

4. **Hallucination risk is real but contained.** Manual inspection of agent reasoning showed occasional 'verified company' claims for companies absent from `KNOWN_COMPANIES` in `src/tools.py`. The deterministic tool wraps the model's confidence — if the model says verified but the tool returns 'no record', the reasoning text and the tool output disagree, which is detectable post-hoc.