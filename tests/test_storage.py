"""Pytest cases for storage.py (the sidebar history store)."""
import storage


def _result(action="caution", score=40):
    return {"action": action, "trust_score": score, "red_flags": ["urgency pressure"],
            "risk_breakdown": {"financial": 70}, "reasoning": "because",
            "neighbor_fraud_rate": 0.6, "features": {"urgency_keywords": 2}}


def test_round_trip(tmp_path):
    p = str(tmp_path / "t.db")
    rid = storage.save_result("some posting text", _result(), [{"label": 1}], path=p)
    got = storage.get_result(rid, path=p)
    assert got["posting"] == "some posting text"
    assert got["result"]["red_flags"] == ["urgency pressure"]
    assert got["result"]["features"] == {"urgency_keywords": 2}
    assert got["neighbors"] == [{"label": 1}]


def test_newest_first_and_filters(tmp_path):
    p = str(tmp_path / "t.db")
    storage.save_result("remote python job", _result("safe", 90), path=p)
    storage.save_result("pay a fee 100% guaranteed", _result("avoid", 10), path=p)
    rows = storage.list_history(path=p)
    assert rows[0]["verdict"] == "avoid"
    assert len(storage.list_history(verdict="safe", path=p)) == 1
    assert len(storage.list_history(text_filter="python", path=p)) == 1
    # % must be matched literally, not as a wildcard
    assert len(storage.list_history(text_filter="100%", path=p)) == 1
    assert len(storage.list_history(text_filter="%", path=p)) == 1


def test_cap_drops_oldest(tmp_path):
    p = str(tmp_path / "t.db")
    for i in range(storage.MAX_ROWS + 5):
        storage.save_result(f"posting {i}", _result(), path=p)
    rows = storage.list_history(path=p)
    assert len(rows) == storage.MAX_ROWS
    assert rows[-1]["posting_text"] == "posting 5"


def test_delete_and_clear(tmp_path):
    p = str(tmp_path / "t.db")
    a = storage.save_result("a", _result(), path=p)
    storage.save_result("b", _result(), path=p)
    storage.delete_one(a, path=p)
    assert len(storage.list_history(path=p)) == 1
    storage.clear_all(path=p)
    assert storage.list_history(path=p) == []


def test_corrupt_file_is_recreated(tmp_path):
    p = tmp_path / "t.db"
    p.write_bytes(b"this is definitely not a sqlite database" * 50)
    assert storage.list_history(path=str(p)) == []
    assert (tmp_path / "t.db.corrupt").exists()
    storage.save_result("works after recovery", _result(), path=str(p))
    assert len(storage.list_history(path=str(p))) == 1


def test_time_ago():
    assert storage.time_ago("not a date") == ""
    assert storage.time_ago(storage._now()) == "just now"
