# Study sheet — `src/pipeline.py`

The single function `analyze_job(text)` is the only public entry
point for the Streamlit app at STEP 15. Everything else in the
project is a building block; this is where they get glued together.

## Flow inside `analyze_job(text)`

1. **`init_db(db_path)`** — make sure SQLite tables exist (cheap;
   no-op if they already do).
2. **`extract_all(text)`** — the 7 heuristic features from
   `src/features.py`.
3. **`vs.neighbor_fraud_rate(text, k=5)`** — fraction of the 5 nearest
   training postings that were fraud. One float in [0, 1].
4. **`agent.invoke({"input": text})`** — the LangChain ReAct agent
   from `src/agent.py` calls tools, reasons, and writes a Final
   Answer (string).
5. **`_parse_json(answer)`** — pull the JSON object out of the
   model's output. Strips markdown fences, walks balanced braces.
6. **Defensive retry** — if the parse fails, ask the agent again
   with an explicit "your previous output was invalid; output ONLY
   the JSON now" prompt. If that also fails, return
   `{"error": ...}` so the UI renders an apology, not a crash.
7. **Enrich** — attach `neighbor_fraud_rate` and the heuristic
   features to the returned dict so the Streamlit UI can show them.
8. **`log_prediction(...)`** — write the row to SQLite. Wrapped in
   try/except because logging is best-effort: a DB hiccup must not
   eat a successful prediction.

## Module-level caches

`_AGENT` and `_VS` are module-level singletons set on first call.
This matters for Streamlit, which reruns the script top-to-bottom
on every interaction — without caching, every "Analyze" click would
reload the LLM and the embedder. With caching, the first click
takes ~30 s (model + embedder boot) and subsequent clicks reuse the
warm objects.

`@st.cache_resource` would be the more idiomatic Streamlit choice,
but using globals here keeps `pipeline.py` independent of Streamlit
— so the same module also works from CLI scripts and notebooks.

## Sample viva Q&A

**Q: Why is the JSON parser separate from the agent? Couldn't the
agent just return a dict?**
A: LangChain's ReAct agent returns plain text in `output` because
the underlying LLM emits text. A "Final Answer" line in ReAct is a
string the parser then has to extract from. We could have used a
structured-output agent (function calling) but ReAct keeps the
prompt simple and works on any chat LLM, including the local Phi-3
once we're on the fine-tuned path. So the parsing is the cost of
keeping the agent backend-agnostic.

**Q: What happens when the model produces a different schema each
call?**
A: That's what the retry path catches. First call fails the parser,
second call gets an explicit "fix the JSON" instruction. If the
second call also fails, we surface the raw text in
`result["raw_output"]` so the UI can show it to the user. We never
silently produce a wrong answer — either we have valid JSON or we
say so.

**Q: Why log to SQLite inside `analyze_job` instead of in the UI?**
A: One source of truth. The Streamlit "recent analyses" panel reads
from `predictions`. The eval scripts also write to `eval_runs`.
Putting the log call in the pipeline means *every* path that
produces a prediction logs it — CLI tests, notebook calls,
Streamlit, future API endpoints. If logging lived in the UI we'd
miss everything that doesn't go through the UI.
