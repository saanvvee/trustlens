# Study sheet — `src/baselines.py`

Four baselines satisfy the rubric's "improvement over baselines"
requirement. Each writes one JSONL of predictions on the val set so
`evaluate.py` (STEP 14) can score them all the same way.

| File | Approach | Why include it |
|---|---|---|
| `baseline_phi3_zeroshot.jsonl` | B1: Phi-3-mini, schema-only prompt | Lower bound for the LLM family — no examples, just instructions |
| `baseline_phi3_fewshot.jsonl` | B2: Phi-3-mini + 3 teacher-labelled examples | Shows ICL gives Phi-3 a free boost without fine-tuning |
| `baseline_llama_fewshot.jsonl` | B3: Llama-3.3-70B-Instruct + 3 examples (HF API) | Strong-LLM reference — can a fine-tuned 3.8B match a 70B? |
| `baseline_distilbert.jsonl` | B4: DistilBERT classifier on the binary fraud label | Non-LLM baseline (rubric explicitly demands this) |

## Shared helpers

### `_load_examples(n=3)`
Reads `data/labeled/train.jsonl` (the teacher labels from STEP 8),
samples `n` rows with `random.seed(42)` for reproducibility. B2 and B3
both use the same 3 examples.

### `_build_messages(job_text, features, nfr, examples=None)`
Composes the chat-template messages list. Always starts with
`SYSTEM_PROMPT` (imported from `label_generator`), then for few-shot
inserts each example as alternating `user`/`assistant` turns
(reconstructing the user prompt with `build_user_prompt`, replaying
the teacher's JSON as the assistant turn), then the real query at
the end. Same shape works for both Phi-3 chat template and
Llama chat completion.

### `_write(name, rows)` — newline-delimited JSON to `data/labeled/`.

## The four baselines

### `baseline_phi3(name, fewshot=False)` — B1 / B2
Loads `microsoft/Phi-3-mini-4k-instruct` in fp16 with
`device_map="auto"` so it lands on GPU if available. Sets
`HF_HUB_OFFLINE=1` to avoid SSL revalidation issues. Iterates val
rows, builds messages, applies the chat template, generates up to
512 new tokens deterministically (`do_sample=False`), parses the
JSON. Same function powers both B1 (no examples) and B2 (3 examples).

### `baseline_llama_fewshot()` — B3
Async function. Same prompt shape as B2 but calls Llama-3.3-70B via
`AsyncInferenceClient.chat_completion`. Reuses
`CONCURRENCY=3` from `label_generator` to respect free-tier limits.
On exception per row, the function returns an empty parse so
`evaluate.py` can count it as a JSON-validity failure.

### `baseline_distilbert()` — B4
The non-LLM baseline. Trains `distilbert-base-uncased` for 1 epoch
on `(job_text, fraudulent)` pairs from train.csv, predicts on val,
and **maps the binary fraud probability into the trust-assessment
schema**:

- `trust_score = 100 * (1 - p_fraud)` (rounded to int)
- `action = "avoid"` if `p_fraud > 0.5` else `"safe"`
- `red_flags = []`, `risk_breakdown = {}`, `reasoning = "DistilBERT
  fraud probability X.XX"`

The empty `red_flags` and `risk_breakdown` are deliberate: B4 has no
way to produce them, and exposing that absence is *exactly the point*
— it shows what the LLM gives us beyond a binary classifier.

## Sample viva Q&A

**Q: Why both B1 (zero-shot) and B2 (few-shot) on the same Phi-3?**
A: To isolate the contribution of in-context learning vs the model's
own instruction-following. If B2 ≈ B1, then the teacher labels
aren't helping ICL and we expect fine-tuning to give a bigger win.
If B2 is much better than B1, ICL is doing real work, and we expect
fine-tuning to deliver an even larger improvement than the gap
between B1 and B2.

**Q: Why DistilBERT specifically for B4 and not logistic regression?**
A: The rubric calls for a "non-LLM classifier baseline." DistilBERT
on the raw `job_text` is the strongest baseline you can get in this
family without ballooning the project. A logistic regression on
TF-IDF would be even simpler but would score so much worse it would
be unfair — we want the fine-tuned LLM's win to be over a *real*
non-LLM baseline, not a strawman.

**Q: B4 predicts trust_score = 100 * (1 - p_fraud). Isn't that
arbitrary?**
A: It's a deliberate mapping. DistilBERT only outputs a probability;
to compare it to the LLM's structured JSON we have to put it on the
same axes. The mapping is monotone (higher probability of fraud →
lower trust score) so all our regression metrics (MAE, Pearson r) on
trust_score still make sense. The empty `red_flags` and
`risk_breakdown` are not bugs — they're features. They show what
B4 *can't* do, which is most of the system.
