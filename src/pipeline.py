"""End-to-end inference pipeline for TrustLens."""

from src.db import DEFAULT_PATH, init_db, log_prediction
from src.features import extract_all
from src.label_generator import _parse_json
from src.vector_store import VectorStore

_AGENT = None
_VS = None


def _agent():
    global _AGENT
    if _AGENT is None:
        from src.agent import build_agent
        _AGENT = build_agent()
    return _AGENT


def _vs():
    global _VS
    if _VS is None:
        _VS = VectorStore()
    return _VS


# 🔥 YOUR HYBRID SCORING SYSTEM
def trustlens_output(pred, features):
    red_flags = []

    if features.get("urgency_keywords", 0) > 0:
        red_flags.append("urgency pressure")

    if features.get("payment_upfront_keywords", 0) > 0:
        red_flags.append("upfront payment request")

    if features.get("suspicious_contact", 0) > 0:
        red_flags.append("suspicious contact details")

    if features.get("missing_company_signals", 0) > 0:
        red_flags.append("missing company info")

    if features.get("salary_range_pattern", 0) > 0:
        red_flags.append("unrealistic salary")

    # 🔥 weighted risk scoring
    risk_score = 0

    if pred == 1:
        risk_score += 30
    elif pred == 0.5:
        risk_score += 15

    risk_score += min(len(red_flags) * 8, 40)

    # avoid extreme 0 unless absolutely worst
    trust_score = max(5, 100 - risk_score)

    # calibrated action
    if trust_score < 30:
        action = "Avoid"
    elif trust_score < 65:
        action = "Caution"
    else:
        action = "Safe"

    return {
        "trust_score": trust_score,
        "red_flags": red_flags,
        "risk_breakdown": {
            "financial": 70 if "upfront payment request" in red_flags else 30,
            "legitimacy": 80 if pred >= 0.5 else 30,
            "data_privacy": 60 if "suspicious contact details" in red_flags else 20
        },
        "action": action,
        "reasoning": "Hybrid decision using LLM signal + heuristic risk features"
    }


def analyze_job(job_text: str, db_path: str = DEFAULT_PATH) -> dict:
    """Main pipeline function"""

    # ✅ ensure DB exists
    init_db(db_path)

    # 🔹 extract features
    feats = extract_all(job_text)

    # 🔹 vector similarity signal
    nfr = _vs().neighbor_fraud_rate(job_text, k=5)

    # 🔹 call LLM
    try:
        result = _agent().invoke({"input": job_text})
        answer = result.get("output", "")
    except Exception as e:
        return {"error": f"agent failed: {type(e).__name__}: {e}"}

    # 🔹 parse LLM output
    parsed = _parse_json(answer)

    # 🔁 retry if JSON failed
    if parsed is None:
        try:
            retry = _agent().invoke({
                "input": f"{job_text}\n\nIMPORTANT: Output ONLY valid JSON."
            })
            parsed = _parse_json(retry.get("output", ""))
        except Exception:
            parsed = None

    # ❌ still failed
    if parsed is None:
        return {
            "error": "model did not produce valid JSON",
            "raw_output": answer[:500]
        }

    # 🔥 derive prediction from LLM output
    action = parsed.get("action", "").lower()

    if action == "avoid":
        pred = 1
    elif action == "caution":
        pred = 0.5
    else:
        pred = 0

    # 🔥 apply YOUR scoring system
    scored = trustlens_output(pred, feats)

    # 🔥 keep LLM reasoning if available
    scored["reasoning"] = parsed.get("reasoning", scored["reasoning"])

    # 🔹 attach signals for UI
    scored["neighbor_fraud_rate"] = nfr
    scored["features"] = feats

    # 🔹 log to DB (safe)
    try:
        log_prediction(
            job_text=job_text,
            trust_score=int(scored.get("trust_score", 0)),
            action=scored.get("action", "caution"),
            reasoning=scored.get("reasoning", ""),
            red_flags=scored.get("red_flags", []),
            risk_breakdown=scored.get("risk_breakdown", {}),
            path=db_path,
        )
    except Exception as e:
        scored["_db_warning"] = str(e)

    return scored