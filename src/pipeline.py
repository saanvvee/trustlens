"""End-to-end inference pipeline for TrustLens.

`analyze_job(text)` is the single function called by the Streamlit
front-end at STEP 15. It runs:

1. ``extract_all`` — heuristic features.
2. ``neighbor_fraud_rate`` — ChromaDB kNN signal.
3. ``agent.invoke`` — LangChain ReAct over the 4 tools.
4. JSON parse the agent's Final Answer (with one retry on failure).
5. ``log_prediction`` — write the row to SQLite.
6. Return the parsed dict (or an error dict if the model misbehaved).
"""
from src.db import DEFAULT_PATH, init_db, log_prediction
from src.features import extract_all
from src.label_generator import _parse_json
from src.vector_store import VectorStore

_AGENT = None  # cached so we don't reload the LLM on every Streamlit call
_VS = None


def _agent():
    global _AGENT
    if _AGENT is None:
        from src.agent import build_agent
        _AGENT = build_agent()  # demo path: HF API
    return _AGENT


def _vs():
    global _VS
    if _VS is None:
        _VS = VectorStore()
    return _VS


def analyze_job(job_text: str, db_path: str = DEFAULT_PATH) -> dict:
    """Assess one posting and log the result. Returns the prediction dict.

    On model misbehaviour (non-JSON output) we retry once with a
    repair prompt. If that still fails we return ``{"error": ...}``
    so the UI can render an apology instead of crashing.
    """
    init_db(db_path)

    feats = extract_all(job_text)
    nfr = _vs().neighbor_fraud_rate(job_text, k=5)

    try:
        result = _agent().invoke({"input": job_text})
        answer = result.get("output", "")
    except Exception as e:
        return {"error": f"agent failed: {type(e).__name__}: {e}"}

    parsed = _parse_json(answer)
    if parsed is None:
        # one retry — ask the model to repair its own output
        try:
            retry = _agent().invoke({"input": (
                f"{job_text}\n\nIMPORTANT: your previous Final Answer was "
                "not valid JSON. Output ONLY a single JSON object now."
            )})
            parsed = _parse_json(retry.get("output", ""))
        except Exception:
            parsed = None

    if parsed is None:
        return {"error": "model did not produce valid JSON",
                "raw_output": answer[:500]}

    # enrich with our deterministic signals so the UI can show them
    parsed["neighbor_fraud_rate"] = nfr
    parsed["features"] = feats

    try:
        log_prediction(
            job_text=job_text,
            trust_score=int(parsed.get("trust_score", 0)),
            action=parsed.get("action", "caution"),
            reasoning=parsed.get("reasoning", ""),
            red_flags=parsed.get("red_flags", []),
            risk_breakdown=parsed.get("risk_breakdown", {}),
            path=db_path,
        )
    except Exception as e:
        # logging is best-effort — don't fail the request on a DB hiccup
        parsed["_db_warning"] = str(e)

    return parsed
