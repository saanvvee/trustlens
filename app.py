"""Streamlit frontend for TrustLens.

Run from project root:
    streamlit run app.py

Calls ``src.pipeline.analyze_job`` for every assessment. Backend
agent uses Llama-3.3-70B via HF Inference API by default; swap to
the fine-tuned Phi-3 LoRA path by passing an ``adapter_path`` in
``src/agent.py``'s ``build_agent``.
"""
import streamlit as st

# st.set_page_config MUST be the very first Streamlit call.
# Anything else above this (st.write, etc.) crashes the app.
st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="wide")

import os
import json
import plotly.graph_objects as go

from src.db import get_recent_predictions, init_db
from src.pipeline import analyze_job
from src.vector_store import VectorStore

# absolute path so the DB lands in the project root no matter what
# directory Streamlit was launched from
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trustlens.db")

# create the SQLite tables on app startup so the sidebar query
# doesn't crash on a fresh install
init_db(DB_PATH)

ACTION_COLORS = {"avoid": "#d9534f", "caution": "#f0ad4e", "safe": "#5cb85c"}


def _gauge(score):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Trust score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3a7bd5"},
            "steps": [
                {"range": [0, 30], "color": "#f8d7da"},
                {"range": [30, 70], "color": "#fff3cd"},
                {"range": [70, 100], "color": "#d4edda"},
            ],
        },
    )).update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))


def _risk_bars(rb):
    keys = ["financial", "legitimacy", "data_privacy"]
    vals = [int(rb.get(k, 0)) for k in keys]
    return go.Figure(go.Bar(
        x=vals, y=keys, orientation="h",
        marker={"color": ["#d9534f" if v > 60 else "#f0ad4e" if v > 30 else "#5cb85c"
                          for v in vals]},
    )).update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"range": [0, 100], "title": "risk (higher = worse)"},
    )


@st.cache_resource
def _vs():
    return VectorStore()


st.title("TrustLens — spot scam job postings")
st.caption("Paste a job posting; get a structured trust assessment with reasoning.")

with st.sidebar:
    st.header("Recent analyses")
    for r in get_recent_predictions(limit=10, path=DB_PATH):
        with st.container(border=True):
            st.markdown(f"**{r['action'].upper()}** · score {r['trust_score']}")
            st.caption(r["job_text"][:120] + ("…" if len(r["job_text"]) > 120 else ""))

    st.header("Similar past postings")
    st.caption("Top 3 most similar from the training set (live update on each analysis).")
    sim_placeholder = st.empty()


posting = st.text_area(
    "Job posting text",
    height=240,
    placeholder="Paste the full posting here (title, description, salary, contact email)...",
)

if st.button("Analyze", type="primary", disabled=len(posting) < 50):
    with st.spinner("Calling agent (this can take 30–60s on free tier)..."):
        result = analyze_job(posting, db_path=DB_PATH)

    if "error" in result:
        st.error(f"Couldn't get a clean assessment: {result['error']}")
        if "raw_output" in result:
            st.code(result["raw_output"])
    else:
        action = result.get("action", "caution")
        score = int(result.get("trust_score", 50))
        st.markdown(
            f"<h2 style='color:{ACTION_COLORS.get(action, '#666')}'>"
            f"Action: {action.upper()}</h2>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(_gauge(score), use_container_width=True)
        with c2:
            st.subheader("Risk breakdown")
            st.plotly_chart(_risk_bars(result.get("risk_breakdown", {})),
                            use_container_width=True)

        st.subheader("Red flags")
        flags = result.get("red_flags", [])
        if flags:
            st.markdown(" ".join(
                f"<span style='background:#f8d7da;border-radius:12px;"
                f"padding:4px 10px;margin:3px;display:inline-block'>{f}</span>"
                for f in flags
            ), unsafe_allow_html=True)
        else:
            st.caption("No red flags surfaced.")

        with st.expander("Reasoning", expanded=True):
            st.write(result.get("reasoning", "—"))

        with st.expander("Heuristic features + neighbor fraud rate"):
            st.json({"neighbor_fraud_rate": result.get("neighbor_fraud_rate"),
                     "features": result.get("features")})

        # update the sidebar similar-postings panel
        try:
            sims = _vs().query(posting, k=3)
            with sim_placeholder.container():
                for s in sims:
                    label_emoji = "🚨" if s["label"] == 1 else "✅"
                    st.markdown(f"{label_emoji} **dist {s['distance']:.2f}**")
                    st.caption(s["text"][:160] + "…")
        except Exception as e:
            sim_placeholder.warning(f"chroma query failed: {e}")
