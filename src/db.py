"""SQLite logging for predictions, evaluation runs, and error reviews.

Stdlib sqlite3 only — no ORM. Three tables, six functions, and a
DEFAULT_PATH constant.

JSON blobs (red_flags, risk_breakdown) are stored as TEXT. SQLite has
no native JSON type but we never filter on the JSON contents — the
Streamlit UI reads the row and parses the JSON itself.

Connections are opened and closed per call. That's wasteful at scale
but keeps the code thread-safe for Streamlit (each user session is
its own thread) and means there's no shared connection to forget.
"""
import json
import sqlite3
from datetime import datetime, timezone

DEFAULT_PATH = "/Users/saanveesharma/trustlens/trustlens.db"

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_text TEXT NOT NULL,
    trust_score INTEGER NOT NULL,
    action TEXT NOT NULL,
    reasoning TEXT,
    red_flags_json TEXT,
    risk_breakdown_json TEXT,
    created_at TEXT NOT NULL
)"""

CREATE_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    rouge_l REAL,
    f1 REAL,
    mae REAL,
    json_validity REAL,
    run_at TEXT NOT NULL
)"""

CREATE_ERRORS = """
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    error_category TEXT,
    notes TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
)"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(path: str = DEFAULT_PATH) -> None:
    """Create the three tables if they don't already exist."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(CREATE_PREDICTIONS)
        conn.execute(CREATE_EVAL_RUNS)
        conn.execute(CREATE_ERRORS)
        conn.commit()
    finally:
        conn.close()


def log_prediction(job_text, trust_score, action, reasoning,
                   red_flags, risk_breakdown, path=DEFAULT_PATH):
    """Insert one prediction row. Returns the new row's id."""
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO predictions (job_text, trust_score, action, reasoning,"
            " red_flags_json, risk_breakdown_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_text, int(trust_score), action, reasoning,
             json.dumps(red_flags), json.dumps(risk_breakdown), _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def log_eval_run(model_name, rouge_l, f1, mae, json_validity, path=DEFAULT_PATH):
    """Insert one evaluation-run row. Returns the new row's id."""
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO eval_runs (model_name, rouge_l, f1, mae, json_validity, run_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (model_name, float(rouge_l), float(f1), float(mae),
             float(json_validity), _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def log_error(prediction_id, error_category, notes="", path=DEFAULT_PATH):
    """Insert one error-review row. Returns the new row's id."""
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO errors (prediction_id, error_category, notes) VALUES (?, ?, ?)",
            (prediction_id, error_category, notes),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_recent_predictions(limit=20, path=DEFAULT_PATH):
    """Most-recent-first list of prediction dicts."""

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        # 🔥 hard-create table here (no dependency on anything else)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_text TEXT NOT NULL,
            trust_score INTEGER NOT NULL,
            action TEXT NOT NULL,
            reasoning TEXT,
            red_flags_json TEXT,
            risk_breakdown_json TEXT,
            created_at TEXT NOT NULL
        )
        """)
        conn.commit()

        rows = conn.execute(
            "SELECT id, job_text, trust_score, action, reasoning,"
            " red_flags_json, risk_breakdown_json, created_at"
            " FROM predictions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()

    return [{
        "id": r["id"],
        "job_text": r["job_text"],
        "trust_score": r["trust_score"],
        "action": r["action"],
        "reasoning": r["reasoning"],
        "red_flags": json.loads(r["red_flags_json"] or "[]"),
        "risk_breakdown": json.loads(r["risk_breakdown_json"] or "{}"),
        "created_at": r["created_at"],
    } for r in rows]