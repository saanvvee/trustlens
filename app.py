
import streamlit as st

# st.set_page_config MUST be the very first Streamlit call.
# Anything else above this (st.write, etc.) crashes the app.
st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="wide")

import os
import html
import plotly.graph_objects as go

import storage
import styles
from src.db import init_db
from src.pipeline import analyze_job
from src.vector_store import VectorStore

# absolute path so the DB lands in the project root no matter what
# directory Streamlit was launched from (storage.py picks a safe fallback
# for read-only hosts)
DB_PATH = storage.DB_PATH
init_db(DB_PATH)  # the pipeline logs raw predictions here too

# verdict -> (label shown in the UI, colour, headline)
VERDICTS = {
    "safe": ("SAFE", "#6fcf97", "Looks legitimate."),
    "caution": ("CAUTION", "#e9b44c", "Proceed with caution."),
    "avoid": ("SCAM", "#ef6b6b", "Likely a scam — avoid."),
}
FILTER_OPTIONS = {"All": "all", "Safe": "safe", "Caution": "caution", "Scam": "avoid"}

EXAMPLES = {
    "Scam example": (
        "URGENT HIRING!!! Work from home data entry. Earn $5,000 per week, no experience "
        "needed. Limited slots, apply immediately. A refundable registration fee of $99 "
        "is required to receive your starter kit. Send your full name, SSN and bank "
        "details to quickhire.jobs2024@gmail.com. We will contact you on WhatsApp."
    ),
    "Legit example": (
        "Software Engineer II — Payments, Acme Corp (Bengaluru, hybrid). We're looking "
        "for an engineer with 3+ years of experience in Python or Java to build and "
        "maintain our payment-processing services. Compensation: ₹28–36 LPA plus equity "
        "and health insurance. The process: recruiter call, two technical interviews, "
        "and a team-fit conversation. Apply via careers.acmecorp.com; questions to "
        "recruiting@acmecorp.com."
    ),
}

styles.inject()


# ---------- charts ----------
# displayModeBar False hides plotly's camera/zoom toolbar
PLOT_CONFIG = {"displayModeBar": False}


def _gauge(score, color):
    """Slim arc with the score in the middle."""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 46, "color": "#ece9e1", "family": "Instrument Serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(0,0,0,0)",
                     "tickfont": {"color": "#8b8d93", "size": 10}, "nticks": 3},
            "bar": {"color": color, "thickness": 0.18},
            "bgcolor": "rgba(236,233,225,0.06)",
            "borderwidth": 0,
        },
    )).update_layout(height=190, margin=dict(l=30, r=30, t=20, b=0),
                     paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ece9e1"})


def _risk_bars(rb):
    """Three thin horizontal bars."""
    keys = ["financial", "legitimacy", "data_privacy"]
    vals = [int(rb.get(k, 0)) for k in keys]
    return go.Figure(go.Bar(
        x=vals, y=[k.replace("_", " ") for k in keys], orientation="h",
        marker={"color": ["#ef6b6b" if v > 60 else "#e9b44c" if v > 30 else "#6fcf97"
                          for v in vals]},
        text=vals, textposition="outside", textfont={"color": "#8b8d93", "size": 11},
        cliponaxis=False,
    )).update_layout(
        height=170, bargap=0.6, margin=dict(l=0, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 100], "visible": False},
        yaxis={"autorange": "reversed", "tickfont": {"color": "#8b8d93", "size": 12}},
        font={"color": "#ece9e1"},
    )


@st.cache_resource
def _vs():
    return VectorStore()


# ---------- callbacks (run before the page reruns) ----------
def _load_example(key):
    st.session_state["posting"] = EXAMPLES[key]


def _open_item(row_id):
    """Reload a saved result into the main panel — no model call."""
    saved = storage.get_result(row_id)
    if saved:
        st.session_state["current"] = saved
        st.session_state["posting"] = saved["posting"]


def _delete_item(row_id):
    storage.delete_one(row_id)
    cur = st.session_state.get("current")
    if cur and cur.get("id") == row_id:
        st.session_state["current"] = None


def _clear_history():
    storage.clear_all()
    st.session_state["current"] = None
    st.session_state["confirm_clear"] = False


st.session_state.setdefault("current", None)
st.session_state.setdefault("confirm_clear", False)


# ---------- sidebar: persistent history ----------
with st.sidebar:
    st.markdown("<div class='brand'>TrustLens</div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow' style='margin-top:.2rem'>History</div>",
                unsafe_allow_html=True)

    search = st.text_input("Search", placeholder="Search postings…",
                           label_visibility="collapsed")
    choice = st.radio("Verdict", list(FILTER_OPTIONS), horizontal=True,
                      label_visibility="collapsed")
    rows = storage.list_history(search, FILTER_OPTIONS[choice])

    if not rows:
        st.caption("Nothing here yet." if not search and choice == "All"
                   else "No matches.")

    for r in rows:
        label, color, _ = VERDICTS.get(r["verdict"], VERDICTS["caution"])
        preview = r["posting_text"][:60] + ("…" if len(r["posting_text"]) > 60 else "")
        st.markdown(
            f"<div class='hist' style='--v:{color}'>"
            f"<div class='row1'><span class='badge'>{label}</span>"
            f"<span class='score'>{r['trust_score']}"
            f" <span class='ago'>· {storage.time_ago(r['timestamp'])}</span></span></div>"
            f"<div class='prev'>{html.escape(preview)}</div></div>",
            unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.button("Open", key=f"open{r['id']}", on_click=_open_item, args=(r["id"],))
        b2.button("Delete", key=f"del{r['id']}", on_click=_delete_item, args=(r["id"],))

    # clear all, with a confirm step
    if rows or st.session_state["confirm_clear"]:
        st.markdown("<div class='hist'></div>", unsafe_allow_html=True)
        if not st.session_state["confirm_clear"]:
            if st.button("Clear history"):
                st.session_state["confirm_clear"] = True
                st.rerun()
        else:
            st.caption("Delete the whole history? This can't be undone.")
            c1, c2 = st.columns(2)
            c1.button("Yes, clear", on_click=_clear_history)
            if c2.button("Cancel"):
                st.session_state["confirm_clear"] = False
                st.rerun()


# ---------- main: hero ----------
st.markdown(
    "<div class='eyebrow'>Job posting scam detector</div>"
    "<div class='display'>Is this job<br><em>real?</em></div>"
    "<p class='lede'>Paste a job posting and get an explainable trust assessment — "
    "score, red flags and reasoning — from heuristics, vector search and an LLM agent.</p>"
    "<div class='steps'>"
    "<div class='step'><span class='n'>(01)</span><span class='t'>Features</span>"
    "<span class='d'>Scam signals pulled from the text</span></div>"
    "<div class='step'><span class='n'>(02)</span><span class='t'>Retrieval</span>"
    "<span class='d'>Nearest known postings from a Chroma index</span></div>"
    "<div class='step'><span class='n'>(03)</span><span class='t'>Agent</span>"
    "<span class='d'>An LLM reasons over both and returns structured JSON</span></div>"
    "</div><hr class='rule'>",
    unsafe_allow_html=True,
)

st.markdown("<div class='eyebrow'>Try it</div>", unsafe_allow_html=True)
ex_cols = st.columns([1, 1, 3])
for col, key in zip(ex_cols, EXAMPLES):
    col.button(key, on_click=_load_example, args=(key,), use_container_width=True)

posting = st.text_area(
    "Job posting text",
    height=200,
    key="posting",
    label_visibility="collapsed",
    placeholder="Paste the full posting here (title, description, salary, contact email)…",
)

if st.button("Analyze posting", type="primary", disabled=len(posting) < 50):
    st.session_state["current"] = None
    with st.status("Analyzing posting…", expanded=True) as status:
        st.write("Extracting features, retrieving similar postings and running the agent. "
                 "This can take 30–60s on the free tier.")
        result = analyze_job(posting, db_path=DB_PATH)
        if "error" in result:
            status.update(label="Analysis failed", state="error", expanded=True)
        else:
            status.update(label="Analysis complete", state="complete", expanded=False)

    if "error" in result:
        st.error(f"Couldn't get a clean assessment: {result['error']}")
        if "raw_output" in result:
            st.code(result["raw_output"])
    else:
        # similar postings are stored with the result so reopening it later
        # doesn't need the vector store
        neighbors = []
        try:
            for s in _vs().query(posting, k=3):
                neighbors.append({"label": int(s["label"]),
                                  "distance": float(s["distance"]),
                                  "text": s["text"][:200]})
        except Exception as e:
            st.warning(f"Similar-postings lookup failed: {e}")

        result["action"] = str(result.get("action", "caution")).lower()
        row_id = storage.save_result(posting, result, neighbors)
        st.session_state["current"] = {
            "id": row_id, "posting": posting, "neighbors": neighbors,
            "timestamp": storage._now(), "result": result,
        }
        st.rerun()  # redraw so the new item shows up in the sidebar


# ---------- main: result panel ----------
cur = st.session_state["current"]
if cur:
    result, neighbors = cur["result"], cur["neighbors"]
    action = str(result.get("action", "caution")).lower()
    label, color, headline = VERDICTS.get(action, VERDICTS["caution"])
    score = int(result.get("trust_score", 50))

    st.markdown(
        f"<div class='verdict' style='--v:{color}'>"
        f"<div class='tag'>{label} · SCORE {score}/100</div>"
        f"<div class='msg'>{headline}</div>"
        f"<div class='when'>Analyzed {storage.time_ago(cur['timestamp'])}</div></div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.markdown("<div class='eyebrow'>Trust score</div>", unsafe_allow_html=True)
        st.plotly_chart(_gauge(score, color), use_container_width=True, config=PLOT_CONFIG)
    with c2:
        st.markdown("<div class='eyebrow'>Risk breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(_risk_bars(result.get("risk_breakdown", {})),
                        use_container_width=True, config=PLOT_CONFIG)

    st.markdown("<div class='eyebrow' style='margin-top:1.5rem'>Red flags</div>",
                unsafe_allow_html=True)
    flags = result.get("red_flags", [])
    if flags:
        st.markdown("".join(f"<span class='pill'>{html.escape(str(f))}</span>"
                            for f in flags), unsafe_allow_html=True)
    else:
        st.caption("No red flags surfaced.")

    st.markdown("<div class='eyebrow' style='margin-top:1.5rem'>Reasoning</div>"
                f"<div class='reason'>{html.escape(str(result.get('reasoning') or '—'))}</div>",
                unsafe_allow_html=True)

    if neighbors:
        st.markdown("<div class='eyebrow' style='margin-top:2rem'>Similar past postings</div>",
                    unsafe_allow_html=True)
        for n in neighbors:
            tag = "KNOWN SCAM" if n["label"] == 1 else "LEGITIMATE"
            st.markdown(f"<div class='sim'><div class='top'>{tag} · distance "
                        f"{n['distance']:.2f}</div><div class='txt'>"
                        f"{html.escape(n['text'])}…</div></div>", unsafe_allow_html=True)

    with st.expander("Heuristic features + neighbor fraud rate"):
        st.json({"neighbor_fraud_rate": result.get("neighbor_fraud_rate"),
                 "features": result.get("features")})
