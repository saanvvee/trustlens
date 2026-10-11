"""Persistent analysis history for the Streamlit sidebar.

Stdlib sqlite3 only. This is separate from src/db.py (which logs raw
predictions for evaluation) because the sidebar needs enough saved to
re-render a whole result later without calling the model again.

NOTE: Streamlit Community Cloud's disk is ephemeral. History survives
page refreshes and reruns, but NOT a container restart (the app being
rebuilt, redeployed or put to sleep and woken up) — the file is wiped.
"""
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

MAX_ROWS = 100  # keep only the newest 100 analyses

CREATE_HISTORY = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    posting_text TEXT NOT NULL,
    verdict TEXT NOT NULL,          -- safe / caution / avoid
    trust_score INTEGER NOT NULL,
    red_flags TEXT,                 -- json list
    risk_breakdown TEXT,            -- json dict
    reasoning TEXT,
    neighbors TEXT,                 -- json list of similar postings
    extras TEXT                     -- json: neighbor_fraud_rate + features
)"""


def _default_path():
    """Project folder if we can write there, otherwise the system temp dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    folder = here if os.access(here, os.W_OK) else tempfile.gettempdir()
    return os.path.join(folder, "trustlens.db")


DB_PATH = _default_path()


def _connect(path=None):
    """Open the DB and make sure the table exists.

    If the file is missing we just create it. If it's corrupt (sqlite
    raises DatabaseError) we move it aside and start a fresh one rather
    than crashing the whole app.
    """
    path = path or DB_PATH
    for attempt in range(2):
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute(CREATE_HISTORY)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            try:
                conn.close()
            except Exception:
                pass
            if attempt == 1:
                raise
            os.replace(path, path + ".corrupt")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_result(posting, result, neighbors=None, path=None):
    """Store one finished analysis, then trim to MAX_ROWS. Returns the new id."""
    conn = _connect(path)
    try:
        extras = {"neighbor_fraud_rate": result.get("neighbor_fraud_rate"),
                  "features": result.get("features")}
        cur = conn.execute(
            "INSERT INTO history (timestamp, posting_text, verdict, trust_score,"
            " red_flags, risk_breakdown, reasoning, neighbors, extras)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), posting,
             str(result.get("action", "caution")).lower(),
             int(result.get("trust_score", 50)),
             json.dumps(result.get("red_flags", [])),
             json.dumps(result.get("risk_breakdown", {})),
             result.get("reasoning", ""),
             json.dumps(neighbors or []),
             json.dumps(extras, default=str)),
        )
        new_id = cur.lastrowid
        # drop the oldest rows beyond the cap
        conn.execute(
            "DELETE FROM history WHERE id NOT IN"
            " (SELECT id FROM history ORDER BY id DESC LIMIT ?)", (MAX_ROWS,))
        conn.commit()
        return new_id
    finally:
        conn.close()


def list_history(text_filter="", verdict="all", path=None):
    """Newest first. verdict is 'all' or one of safe / caution / avoid."""
    sql = "SELECT id, timestamp, posting_text, verdict, trust_score FROM history"
    where, args = [], []
    if text_filter.strip():
        # escape LIKE wildcards so a typed % or _ is matched literally
        esc = (text_filter.strip().replace("\\", "\\\\")
               .replace("%", "\\%").replace("_", "\\_"))
        where.append("posting_text LIKE ? ESCAPE '\\'")
        args.append(f"%{esc}%")
    if verdict != "all":
        where.append("verdict = ?")
        args.append(verdict)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC"

    conn = _connect(path)
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def get_result(row_id, path=None):
    """Rebuild a saved analysis in the same shape analyze_job returns."""
    conn = _connect(path)
    try:
        r = conn.execute("SELECT * FROM history WHERE id = ?", (row_id,)).fetchone()
    finally:
        conn.close()
    if r is None:
        return None
    extras = json.loads(r["extras"] or "{}")
    return {
        "id": r["id"],
        "posting": r["posting_text"],
        "timestamp": r["timestamp"],
        "neighbors": json.loads(r["neighbors"] or "[]"),
        "result": {
            "action": r["verdict"],
            "trust_score": r["trust_score"],
            "red_flags": json.loads(r["red_flags"] or "[]"),
            "risk_breakdown": json.loads(r["risk_breakdown"] or "{}"),
            "reasoning": r["reasoning"],
            "neighbor_fraud_rate": extras.get("neighbor_fraud_rate"),
            "features": extras.get("features"),
        },
    }


def delete_one(row_id, path=None):
    conn = _connect(path)
    try:
        conn.execute("DELETE FROM history WHERE id = ?", (row_id,))
        conn.commit()
    finally:
        conn.close()


def clear_all(path=None):
    conn = _connect(path)
    try:
        conn.execute("DELETE FROM history")
        conn.commit()
    finally:
        conn.close()


def time_ago(iso_ts):
    """'just now', '5m ago', '2h ago', '3d ago' from a stored UTC timestamp."""
    try:
        then = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return ""
    secs = int((datetime.now(timezone.utc) - then).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
