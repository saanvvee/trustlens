"""Pytest cases for src.label_generator._parse_json.

"""
from src.label_generator import _parse_json, build_user_prompt


def test_parses_clean_json():
    out = _parse_json('{"trust_score": 80, "action": "safe"}')
    assert out == {"trust_score": 80, "action": "safe"}


def test_strips_markdown_fences():
    txt = '```json\n{"trust_score": 12, "action": "avoid"}\n```'
    out = _parse_json(txt)
    assert out["trust_score"] == 12
    assert out["action"] == "avoid"


def test_finds_json_after_leading_prose():
    txt = 'Here is the analysis:\n{"trust_score": 50, "action": "caution"}'
    out = _parse_json(txt)
    assert out["action"] == "caution"


def test_handles_nested_braces():
    txt = '{"trust_score": 30, "risk_breakdown": {"financial": 80, "legitimacy": 60}}'
    out = _parse_json(txt)
    assert out["risk_breakdown"]["financial"] == 80


def test_returns_none_on_garbage():
    assert _parse_json("nope, not json at all") is None


def test_returns_none_on_truncated_json():
    # missing closing brace
    assert _parse_json('{"trust_score": 50, "action":') is None


def test_build_user_prompt_includes_features_and_nfr():
    prompt = build_user_prompt(
        "Earn 5000 weekly!",
        {"urgency_keywords": 2.0, "allcaps_ratio": 0.4},
        neighbor_fraud_rate=0.6,
    )
    assert "Earn 5000 weekly" in prompt
    assert "urgency_keywords: 2.00" in prompt
    assert "neighbor_fraud_rate: 0.60" in prompt
