# Study sheet — `src/tools.py`

This file defines the 4 tools the LangChain ReAct agent (STEP 12) can
call. Each is a plain Python function decorated with `@tool` from
`langchain_core.tools`. Together they replace what would otherwise be
RAG context — the agent reasons over **tool outputs**, not over
retrieved documents.

The decorator does three things automatically:
1. Reads the type hints to build a JSON schema for the tool's args.
2. Reads the docstring as the tool's description (the model uses this
   to decide when to call it).
3. Wraps the function as a `BaseTool` so LangChain can invoke it.

## The 4 tools

### `extract_red_flag_features(job_text)`
Wraps `src.features.extract_all`. Returns the same 7-key float dict
(`urgency_keywords`, `salary_range_pattern`, etc.). The agent calls
this when it wants the cheap heuristic signals before deciding.

### `check_company_legitimacy(name)`
Looks up the name against `KNOWN_COMPANIES` (~30 well-known
employers, hardcoded). Returns `'verified'` or `'no record found'`.

The list is deliberately small. A real production system would query
Crunchbase or SEC EDGAR; for this project we keep it offline and
deterministic so the tool is fast and the result is always
explainable in viva. **Absence is informative, not damning** — most
real companies aren't on a list of 30 names.

### `analyze_salary_realism(role, salary_text, location)`
Parses any numbers out of `salary_text`, takes the max as the posted
upper bound, and compares against `ROLE_BENCHMARKS[role_keyword]`.
Returns one of `'plausible'`, `'suspiciously high'`, `'unparseable'`,
`'no benchmark for role'`.

We only flag *too high*, not *too low*. Legitimate startups underpay,
legitimate offshore roles look low to a US-tuned benchmark, and
flagging low salaries would generate too many false positives.
Threshold is `2× upper-band` — a deliberate "I'd rather miss a
borderline case than flag a real role."

### `check_email_domain(email)`
Splits on `@`, takes the right side, checks against `FREE_DOMAINS`
(gmail/yahoo/etc). Returns `'free webmail'`, `'corporate'`, or
`'no email provided'`. Real companies hire from corporate domains.

### `ALL_TOOLS`
Exported list of all 4 tools so `src/agent.py` (STEP 12) can do
`AgentExecutor(... tools=ALL_TOOLS ...)` without re-importing each.

## Sample viva Q&A

**Q: Why hardcoded company allowlist instead of an API?**
A: Three reasons. (1) Determinism — the same input gives the same
output forever, so eval results are reproducible. (2) Offline — no
external API to break or rate-limit during demo. (3) Defendability —
I can read the list out loud in viva. A real production system
would swap this for an EDGAR / Crunchbase lookup; the *interface*
(`name -> 'verified' | 'no record found'`) doesn't change.

**Q: Why is `location` in `analyze_salary_realism` if you don't use it?**
A: It's in the signature so the agent learns to pass it from the
posting. The current implementation ignores it because building a
correct cost-of-living lookup table is its own project. With the
parameter already plumbed through, swapping in a CoL adjustment
later is a one-line change in this file — no contract change for the
agent.

**Q: How do these tools "replace RAG"?**
A: A RAG system would, on each new posting, retrieve N similar past
postings and stuff their *text* into the prompt. We don't do that —
the closest we come is `neighbor_fraud_rate`, a single number from
ChromaDB. Instead, the agent reasons over **structured signals from
tools**: heuristic features, a verified-company lookup, a salary-
band check, an email-domain class. Each tool output is small,
explainable, and deterministic — the agent's job is to combine them
with the posting text into a structured judgement.
