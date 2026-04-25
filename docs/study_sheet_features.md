# Study sheet — `src/features.py`

This file is the "cheap features" layer. Before we ever ask the LLM
anything, we run the posting text through 7 simple Python functions
that each return one number. Those numbers go into the LLM prompt as a
small numeric block ("here are some signals my code already noticed"),
and they also feed the non-LLM DistilBERT baseline (`baselines.py` at
STEP 9) so the rubric can compare the fine-tuned model against rules.

## What each function does

### Top-of-file constants

- `URGENCY_WORDS` — phrases scammers use to rush you ("urgent",
  "apply now", "act now").
- `PAYMENT_PHRASES` — phrases that imply you'd have to pay to start
  ("registration fee", "starter kit", "deposit").
- `FREE_EMAIL_DOMAINS` — gmail/yahoo/etc. A real company doesn't run
  hiring out of `@gmail.com`.
- `MESSENGER_HINTS` — WhatsApp/Telegram/etc. Real recruiters don't
  move you to WhatsApp.
- `HIGH_PAY_RE` — regex for `$NUMBER/week`, `$NUMBER/day`,
  `$NUMBER/hour` patterns.
- `EMAIL_RE` — regex that catches any email and pulls out its domain.

### `salary_range_pattern(text) -> float`

Looks for `$NUMBER/week`, `$NUMBER/day`, `$NUMBER/hour`. Returns 1.0 if
the rate is suspiciously high (>=$3000/week, >=$500/day, >=$150/hour),
else 0.0. Real engineering salaries are quoted annually so they don't
trip this; "$5000/week from home" does.

### `urgency_keywords(text) -> float`

Lowercases the text, counts how many phrases from `URGENCY_WORDS`
appear. Returns the count as a float so every feature in `extract_all`
has the same dtype.

### `suspicious_contact(text) -> float`

Returns 1.0 if a messenger app is mentioned (WhatsApp etc.) **or** any
email in the text uses a free webmail domain. Otherwise 0.0. We check
messengers first because some scams have both — once we know it's bad
we can stop early.

### `missing_company_signals(text) -> float`

The EDA notebook tagged each posting with `[COMPANY PROFILE]`. If that
tag is absent, the source CSV had no company description (often because
no real company is behind the post). If the tag is present but the
section under it is shorter than 30 characters, we still count that as
missing — a one-line "we sell stuff" is no better than nothing.

### `payment_upfront_keywords(text) -> float`

Same shape as `urgency_keywords`: lowercase + count occurrences of
phrases from `PAYMENT_PHRASES`.

### `allcaps_ratio(text) -> float`

Pulls every word of >=3 letters and returns the fraction that are
entirely uppercase. The 3-letter floor stops legitimate acronyms (IT,
HR, US) from inflating the ratio in normal postings.

### `exclamation_density(text) -> float`

Number of `!` per 1000 characters. Length-normalised so a 3000-char
real posting with two `!`s doesn't look the same as a 200-char ad with
five.

### `extract_all(text) -> dict`

Runs every feature and returns a `name -> float` dict. Coerces `None`
or non-string inputs to `""` so it never raises. This is the only
function the rest of the project calls.

## Sample viva Q&A

**Q: Why hand-coded heuristics in 2026 when you have an LLM?**
A: They're cheap, deterministic, explainable, and they catch the
obvious 80% — which lets the LLM spend its capacity on the ambiguous
cases. They also feed the non-LLM DistilBERT baseline so we can
demonstrate the fine-tuned LLM improves over rules. Without rules, our
baseline section would be LLM-vs-LLM, which the rubric won't accept as
"non-LLM classifier."

**Q: How did you choose the salary thresholds (3000/week, 500/day,
150/hour)?**
A: Real software engineering salaries are quoted annually
(~$60k–$200k), so they don't show up in the regex at all. Anything
quoted at hourly/daily/weekly rates above these thresholds is either a
high-end contractor day-rate (rare in scam-prone job sites) or a
"make $5k/week from home" scam. I stayed conservative — I'd rather
miss a few real scams than false-flag a legitimate role, because we
want the LLM to make the final call.

**Q: Why does `extract_all` return all floats and not booleans?**
A: Two reasons. First, some signals are real counts (urgency, payment),
some are real ratios (allcaps, exclamation), and some are flags
(salary, contact, missing-company). Mixing dtypes makes downstream code
messy. Second, if we feed these into the DistilBERT baseline or any
classical classifier we need numeric input — booleans get cast to
0.0/1.0 anyway, so I do it once here and the type signature is a
single dtype.
