# Study sheet — `src/db.py`

This file is the persistence layer. Three things go into SQLite, all
through this file:

1. Every prediction the Streamlit app makes (`predictions` table).
2. Every evaluation run (`eval_runs` table) — one row per model per
   metrics pass.
3. Every error flagged during error analysis (`errors` table) — one
   row per (prediction, category) pair.

We use **stdlib `sqlite3`** with no ORM. That's deliberate: an ORM
would add a dependency and a layer of magic the user has to defend.
Plain SQL is short, explicit, and shows up clearly in viva.

## What each piece does

### Top-of-file constants
- `DEFAULT_PATH = "trustlens.db"` — relative path used by every
  function unless the caller passes a different one. Tests pass
  `tmp_path` so they don't dirty the real DB.
- `CREATE_PREDICTIONS`, `CREATE_EVAL_RUNS`, `CREATE_ERRORS` —
  multi-line SQL strings. Kept at module scope so the schema is
  readable in one glance.

### `_now()`
Helper that returns the current UTC time as
`"YYYY-MM-DDTHH:MM:SS+00:00"`. Used as the timestamp for both
`predictions.created_at` and `eval_runs.run_at` so they sort
correctly with simple string comparison.

### `init_db(path)`
Opens a connection, runs the three `CREATE TABLE IF NOT EXISTS`
statements, commits, closes. Safe to call any number of times —
existing tables are left alone.

### `log_prediction(...)`
Inserts one row into `predictions`. The two JSON-shaped fields
(`red_flags`, `risk_breakdown`) are serialised with `json.dumps`
before insert. Returns `cur.lastrowid` so the caller (the inference
pipeline at STEP 13) knows the row id and can pass it later to
`log_error` if a reviewer flags it.

### `log_eval_run(...)`
Insert one row into `eval_runs`. Called from `evaluate.py` at
STEP 14, once per model (4 baselines + the fine-tuned model).

### `log_error(prediction_id, error_category, notes)`
Insert one row into `errors`, linked back to a `predictions.id` via
the `prediction_id` foreign key. Called from `error_analyzer.py` at
STEP 16. `notes` defaults to empty so the CLI doesn't have to pass
something every time.

### `get_recent_predictions(limit=20)`
The only `SELECT` in the file. Returns the most recent predictions
as a list of dicts (newest first). The function uses
`conn.row_factory = sqlite3.Row` to access columns by name, then
parses the two JSON blobs back into Python objects. The Streamlit
sidebar at STEP 15 calls this directly.

## Why open-and-close per call?

Streamlit handles each user session in its own thread, and SQLite
connections are not thread-safe by default (you'd need
`check_same_thread=False`). Opening per-call sidesteps the issue
entirely and removes a class of bugs ("did I forget to close the
shared connection?"). The wasted overhead is microseconds per insert
— irrelevant at our traffic.

## Sample viva Q&A

**Q: Why SQLite and not PostgreSQL/MySQL?**
A: SQLite is in stdlib — zero ops, zero install, zero credentials.
The whole project runs on a single laptop; we never need concurrent
write throughput from multiple servers. PostgreSQL would force me to
add a docker-compose and configure a service for no real benefit.

**Q: Why store JSON as TEXT instead of using SQLite's JSON1
extension?**
A: JSON1 lets you query inside the JSON (`json_extract`), but we
never do that — the Streamlit UI reads the row and parses the JSON
in Python. TEXT is portable across SQLite versions, easier to dump
and inspect with `sqlite3` CLI, and one less feature to defend.

**Q: Why is `errors.prediction_id` not `NOT NULL`?**
A: `error_analyzer.py` at STEP 16 also catalogs JSON-validity
failures from baselines that didn't write to `predictions` first.
Those errors don't have a prediction id to link to, so the column
allows NULL. The foreign key is still declared so existing
prediction ids do get validated when used.
