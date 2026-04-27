# Study sheet — `src/label_generator.py`

This file is the **teacher** in the teacher–student setup. We can't
afford to fine-tune Phi-3 on the raw `fraudulent` 0/1 labels — those
labels would only teach the student to predict a single number, not
to *reason* about postings or write structured JSON.

So we use a strong instruction-tuned LLM (Llama-3.3-70B-Instruct on
HuggingFace's free Inference API) to **rewrite** each posting into a
structured trust assessment using a Chain-of-Thought prompt. The
output JSON becomes the supervised signal at STEP 10.

## What each piece does

### Top-of-file constants
- `MODEL = "meta-llama/Llama-3.3-70B-Instruct"` — the teacher. We
  picked the biggest free instruction-tuned model that's
  JSON-disciplined.
- `CONCURRENCY = 3` — async semaphore. The HF free tier is rate-
  limited; running 10 in parallel will get you 429s. 3 is polite.

### `SYSTEM_PROMPT`
The CoT instruction the teacher follows on every posting. Four
explicit numbered steps, then the strict JSON schema. Why steps:
- Step 1 ("list signals") forces the model to surface specific
  evidence before it commits to a score.
- Step 2 ("score each category") prevents one big number that
  averages everything.
- Step 3 ("decide an action") makes the categorical choice the
  *consequence* of the analysis, not a random guess.
- Step 4 ("output ONLY JSON") fights the model's natural tendency
  to add prose.

### `build_user_prompt(job_text, features, neighbor_fraud_rate)`
Composes the user message: the posting itself, the heuristic feature
block from `src/features.py`, and the kNN-derived neighbor fraud
rate from `src/vector_store.py`. The two numeric blocks are what
makes this *not RAG* — we feed numbers and rules, never retrieved
text from past postings.

### `_parse_json(text)`
Models lie. They emit ` ```json ... ``` ` fences, leading prose,
trailing apologies. This function:
1. Strips markdown backticks and the optional `json` language tag.
2. Finds the first `{`.
3. Walks forward, balancing braces, until depth returns to 0.
4. Tries `json.loads` on that span. Returns `None` on failure.

The 7 pytest cases pin down its behaviour for clean JSON, fenced
JSON, prose-prefixed JSON, nested braces, garbage, and truncation.

### `label_one(client, sem, ..., retries=3)`
The async unit of work. Acquires the semaphore, calls
`chat_completion`, parses the response. On exception (rate limit,
network blip), it sleeps `5 * 2**attempt` seconds and retries up to
3 times — exponential backoff. Returns `None` only after the third
failure so a few bad postings don't kill the whole job.

### `run(input_csv, output_path, n, dry_run)`
1. Loads the env, opens the CSV.
2. Stratified-samples 50 / 50 fraud / real up to `n` rows. Why
   50/50: the natural 5 % fraud rate would waste teacher calls on
   easy real postings the student already gets right. We want the
   student to learn what fraud looks like.
3. Builds one async task per row, gathers them with
   `asyncio.as_completed` so completed labels stream in as they
   land (rather than waiting for the slowest one).
4. In dry-run, prints the 5 results as JSON to stdout and exits.
5. Otherwise writes JSON-lines to `data/labeled/train.jsonl`.

### `main()`
Argparse + `asyncio.run`. Defaults to 1000 rows because the HF free
tier won't reliably handle 1500 in one go.

## Sample viva Q&A

**Q: Why teacher–student instead of training Phi-3 directly on the
0 / 1 fraud labels?**
A: The end product has to *write a JSON assessment with reasoning*,
not output a binary. Training Phi-3 on 0 / 1 labels would teach it
to compress everything into one digit. By having a strong teacher
generate full CoT reasoning + structured JSON for each posting, the
student learns to imitate that whole behaviour — reasoning, scoring,
choosing an action, and explaining itself. We're using the labels
we have (0 / 1) to *select* which postings to teach on, not as the
direct supervision signal.

**Q: Why a Chain-of-Thought prompt and not just "give me the JSON"?**
A: Without CoT, instruction-tuned models default to averaging.
They'll produce a vaguely plausible JSON that doesn't reflect the
specific signals in the posting. The numbered four-step prompt
forces the model to (i) surface concrete signals before scoring,
(ii) score each category before deciding action, (iii) commit to a
specific action. The JSON it emits is the *result* of that visible
chain — not an averaged guess. This is the same pattern as the
Self-Consistency / Self-Critique families of papers.

**Q: What if the teacher is wrong on some postings?**
A: Two layers of defence. First, we class-balance the training set
50 / 50, so the student sees roughly the same number of correct
fraud and correct real labels — random teacher noise distributes
across both classes. Second, the original `fraudulent` 0 / 1 column
is preserved in every output row, so at evaluation time we measure
F1-on-action against the *ground truth* label, not against the
teacher's `action`. If the teacher is systematically wrong, the
student's val F1 will surface it.
