
import streamlit as st

# st.set_page_config MUST be the very first Streamlit call.
# Anything else above this (st.write, etc.) crashes the app.
st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="wide")

import os
import html
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

ACTION_COLORS = {"avoid": "#ff6b6b", "caution": "#ffb454", "safe": "#4ade80"}
ACTION_LABELS = {"avoid": "High risk — avoid", "caution": "Proceed with caution",
                 "safe": "Looks legitimate"}

EXAMPLES = {
    "🚨 Scam example": (
        "URGENT HIRING!!! Work from home data entry. Earn $5,000 per week, no experience "
        "needed. Limited slots, apply immediately. A refundable registration fee of $99 "
        "is required to receive your starter kit. Send your full name, SSN and bank "
        "details to quickhire.jobs2024@gmail.com. We will contact you on WhatsApp."
    ),
    "✅ Legit example": (
        "Software Engineer II — Payments, Acme Corp (Bengaluru, hybrid). We're looking "
        "for an engineer with 3+ years of experience in Python or Java to build and "
        "maintain our payment-processing services. Compensation: ₹28–36 LPA plus equity "
        "and health insurance. The process: recruiter call, two technical interviews, "
        "and a team-fit conversation. Apply via careers.acmecorp.com; questions to "
        "recruiting@acmecorp.com."
    ),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, textarea { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; max-width: 1150px; }

.hero { padding: 2.2rem 2rem; border-radius: 20px; margin-bottom: 1.2rem;
  background: radial-gradient(circle at 15% 20%, rgba(124,156,255,.22), transparent 55%),
              radial-gradient(circle at 90% 80%, rgba(74,222,128,.14), transparent 50%), #131a2b;
  border: 1px solid rgba(255,255,255,.07); }
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 .4rem 0; letter-spacing: -.02em;
  background: linear-gradient(90deg, #fff, #9db4ff); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero p { color: #aab3c8; font-size: 1.05rem; margin: 0; max-width: 640px; }

.steps { display: flex; gap: .7rem; flex-wrap: wrap; margin: 1rem 0 1.6rem; }
.step { flex: 1; min-width: 180px; padding: .9rem 1rem; border-radius: 14px;
  background: #131a2b; border: 1px solid rgba(255,255,255,.06); }
.step b { color: #9db4ff; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
.step div { color: #cfd5e5; font-size: .92rem; margin-top: .2rem; }

.verdict { padding: 1.3rem 1.5rem; border-radius: 16px; margin: 1rem 0;
  background: #131a2b; border: 1px solid var(--c); box-shadow: 0 0 32px -12px var(--c); }
.verdict .tag { display: inline-block; padding: .2rem .8rem; border-radius: 999px;
  font-weight: 700; font-size: .78rem; letter-spacing: .08em; color: #0b0f1a; background: var(--c); }
.verdict .msg { font-size: 1.5rem; font-weight: 700; margin-top: .5rem; color: #fff; }

.chip { display: inline-block; margin: 3px 6px 3px 0; padding: 5px 13px; border-radius: 999px;
  background: rgba(255,107,107,.12); border: 1px solid rgba(255,107,107,.4);
  color: #ffb3b3; font-size: .88rem; }
.card { padding: 1rem 1.2rem; border-radius: 14px; background: #131a2b;
  border: 1px solid rgba(255,255,255,.06); color: #cfd5e5; line-height: 1.6; }
.sim { padding: .6rem .8rem; border-radius: 10px; background: #0b0f1a; margin-bottom: .5rem;
  border: 1px solid rgba(255,255,255,.06); }

div.stButton > button[kind="primary"] { border-radius: 12px; font-weight: 600; padding: .6rem 1.6rem;
  background: linear-gradient(90deg, #5b7cff, #8b6cff); border: none; }
div.stButton > button { border-radius: 12px; }
textarea { border-radius: 12px !important; }
[data-testid="stSidebar"] { background: #0e1424; }
</style>
"""


def _gauge(score, color):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 54, "color": "#fff"}},
        title={"text": "Trust score", "font": {"size": 15, "color": "#aab3c8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#566"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(255,107,107,.18)"},
                {"range": [30, 65], "color": "rgba(255,180,84,.18)"},
                {"range": [65, 100], "color": "rgba(74,222,128,.18)"},
            ],
        },
    )).update_layout(height=270, margin=dict(l=30, r=40, t=50, b=10),
                     paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6e9f2"})


def _risk_bars(rb):
    keys = ["financial", "legitimacy", "data_privacy"]
    vals = [int(rb.get(k, 0)) for k in keys]
    return go.Figure(go.Bar(
        x=vals, y=[k.replace("_", " ") for k in keys], orientation="h",
        marker={"color": ["#ff6b6b" if v > 60 else "#ffb454" if v > 30 else "#4ade80"
                          for v in vals]},
        text=vals, textposition="outside",
    )).update_layout(
        height=240, margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 110], "title": "risk (higher = worse)",
               "gridcolor": "rgba(255,255,255,.06)"},
        yaxis={"autorange": "reversed"}, font={"color": "#e6e9f2"},
    )


@st.cache_resource
def _vs():
    return VectorStore()


def _load_example(key):
    st.session_state["posting"] = EXAMPLES[key]


st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='hero'><h1>TrustLens</h1>"
    "<p>Paste a job posting and get an explainable scam-risk assessment — trust score, "
    "red flags and reasoning — from a hybrid of heuristics, vector search and an LLM "
    "agent.</p></div>"
    "<div class='steps'>"
    "<div class='step'><b>1 · Features</b><div>Heuristic scam signals extracted from the text</div></div>"
    "<div class='step'><b>2 · Retrieval</b><div>Nearest known postings from a Chroma index</div></div>"
    "<div class='step'><b>3 · Agent</b><div>LLM reasons over both and returns structured JSON</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🔍 TrustLens")
    st.caption("QLoRA-tuned Phi-3-mini · LangChain · ChromaDB · Groq")
    st.markdown("#### Recent analyses")
    recent = get_recent_predictions(limit=10, path=DB_PATH)
    if not recent:
        st.caption("Nothing yet — run your first analysis.")
    for r in recent:
        with st.container(border=True):
            st.markdown(f"**{r['action'].upper()}** · score {r['trust_score']}")
            st.caption(r["job_text"][:120] + ("…" if len(r["job_text"]) > 120 else ""))

    st.markdown("#### Similar past postings")
    st.caption("Top 3 most similar from the training set.")
    sim_placeholder = st.empty()

st.markdown("##### Try it")
ex_cols = st.columns(len(EXAMPLES) + 1)
for col, key in zip(ex_cols, EXAMPLES):
    col.button(key, on_click=_load_example, args=(key,), use_container_width=True)

posting = st.text_area(
    "Job posting text",
    height=220,
    key="posting",
    label_visibility="collapsed",
    placeholder="Paste the full posting here (title, description, salary, contact email)...",
)

if st.button("Analyze posting", type="primary", disabled=len(posting) < 50):
    with st.spinner("Running the agent — this can take 30–60s on the free tier..."):
        result = analyze_job(posting, db_path=DB_PATH)

    if "error" in result:
        st.error(f"Couldn't get a clean assessment: {result['error']}")
        if "raw_output" in result:
            st.code(result["raw_output"])
    else:
        action = str(result.get("action", "caution")).lower()
        score = int(result.get("trust_score", 50))
        color = ACTION_COLORS.get(action, "#8892a6")
        st.markdown(
            f"<div class='verdict' style='--c:{color}'>"
            f"<span class='tag'>{action.upper()}</span>"
            f"<div class='msg'>{ACTION_LABELS.get(action, action)}</div></div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_gauge(score, color), use_container_width=True)
        with c2:
            st.markdown("##### Risk breakdown")
            st.plotly_chart(_risk_bars(result.get("risk_breakdown", {})),
                            use_container_width=True)

        st.markdown("##### Red flags")
        flags = result.get("red_flags", [])
        if flags:
            st.markdown("".join(f"<span class='chip'>{html.escape(str(f))}</span>"
                                for f in flags), unsafe_allow_html=True)
        else:
            st.caption("No red flags surfaced.")

        st.markdown("##### Reasoning")
        st.markdown(f"<div class='card'>{html.escape(str(result.get('reasoning', '—')))}</div>",
                    unsafe_allow_html=True)

        with st.expander("Heuristic features + neighbor fraud rate"):
            st.json({"neighbor_fraud_rate": result.get("neighbor_fraud_rate"),
                     "features": result.get("features")})

        # update the sidebar similar-postings panel
        try:
            sims = _vs().query(posting, k=3)
            with sim_placeholder.container():
                for s in sims:
                    label_emoji = "🚨" if s["label"] == 1 else "✅"
                    st.markdown(
                        f"<div class='sim'>{label_emoji} <b>dist {s['distance']:.2f}</b><br>"
                        f"<small>{html.escape(s['text'][:160])}…</small></div>",
                        unsafe_allow_html=True)
        except Exception as e:
            sim_placeholder.warning(f"chroma query failed: {e}")
