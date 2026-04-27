"""Pytest cases for src/db.py.

Each test gets a fresh empty database via the tmp_path fixture so
tests don't see each other's rows.
"""
import sqlite3

from src.db import (
    init_db,
    log_prediction,
    log_eval_run,
    log_error,
    get_recent_predictions,
)


def _db(tmp_path):
    """Return a fresh DB path inside tmp_path with all tables created."""
    p = str(tmp_path / "test.db")
    init_db(p)
    return p


def test_init_db_creates_three_tables(tmp_path):
    p = _db(tmp_path)
    conn = sqlite3.connect(p)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    conn.close()
    names = [r[0] for r in rows]
    assert names == ["errors", "eval_runs", "predictions"]


def test_log_prediction_returns_rowid_and_persists(tmp_path):
    p = _db(tmp_path)
    rid = log_prediction(
        job_text="Earn 5000 weekly!",
        trust_score=12,
        action="avoid",
        reasoning="Pay rate is unrealistic.",
        red_flags=["high_pay", "urgency"],
        risk_breakdown={"financial": 90, "legitimacy": 80, "data_privacy": 30},
        path=p,
    )
    assert rid == 1
    rows = get_recent_predictions(limit=10, path=p)
    assert len(rows) == 1
    assert rows[0]["trust_score"] == 12
    assert rows[0]["action"] == "avoid"
    assert rows[0]["red_flags"] == ["high_pay", "urgency"]
    assert rows[0]["risk_breakdown"]["financial"] == 90


def test_log_eval_run_persists(tmp_path):
    p = _db(tmp_path)
    rid = log_eval_run(
        model_name="phi3-trustlens-lora",
        rouge_l=0.42, f1=0.78, mae=8.5, json_validity=0.96, path=p,
    )
    assert rid == 1
    conn = sqlite3.connect(p)
    row = conn.execute("SELECT model_name, f1 FROM eval_runs").fetchone()
    conn.close()
    assert row[0] == "phi3-trustlens-lora"
    assert abs(row[1] - 0.78) < 1e-9


def test_log_error_links_to_prediction(tmp_path):
    p = _db(tmp_path)
    pid = log_prediction(
        job_text="x", trust_score=50, action="caution", reasoning="r",
        red_flags=[], risk_breakdown={}, path=p,
    )
    eid = log_error(prediction_id=pid, error_category="wrong_action",
                    notes="should have been avoid", path=p)
    assert eid == 1
    conn = sqlite3.connect(p)
    row = conn.execute(
        "SELECT prediction_id, error_category FROM errors"
    ).fetchone()
    conn.close()
    assert row[0] == pid
    assert row[1] == "wrong_action"


def test_get_recent_predictions_orders_newest_first(tmp_path):
    p = _db(tmp_path)
    for i in range(5):
        log_prediction(
            job_text=f"posting {i}", trust_score=i * 10, action="caution",
            reasoning="r", red_flags=[], risk_breakdown={}, path=p,
        )
    rows = get_recent_predictions(limit=3, path=p)
    assert len(rows) == 3
    # newest first means descending id
    assert [r["id"] for r in rows] == [5, 4, 3]


def test_get_recent_predictions_handles_empty_db(tmp_path):
    p = _db(tmp_path)
    assert get_recent_predictions(limit=10, path=p) == []
