"""All the custom CSS for the app in one place.

Direction: dark and editorial. Near-black background, one warm cream
accent, big serif headings, hairline borders instead of boxed cards.
Verdict colours (green / amber / red) are the only other colours and
are set per-verdict from app.py through the --v CSS variable.
"""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg: #0a0b0d;
  --text: #ece9e1;
  --mute: #8b8d93;
  --line: rgba(236, 233, 225, 0.12);
  --accent: #e9e4d4;
}

/* ---------- base ---------- */
html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: 'Inter', sans-serif;
  font-weight: 400;
}
.stApp { background: var(--bg); }

/* hide streamlit chrome. The header stays (transparent) because the
   sidebar toggle lives in it on narrow screens. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; padding-bottom: 5rem; max-width: 1000px; }

/* ---------- type ---------- */
.display { font-family: 'Instrument Serif', serif; font-weight: 400;
  font-size: clamp(2.8rem, 8vw, 5.6rem); line-height: 1.0; letter-spacing: -0.02em;
  margin: 1rem 0 0 0; color: var(--text); }
.display em { font-style: italic; color: var(--accent); }
.lede { color: var(--mute); font-size: 1.02rem; line-height: 1.65; max-width: 520px;
  margin: 2.2rem 0 0 0; }
.eyebrow { font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--mute); margin: 0 0 .9rem 0; }
.h2 { font-family: 'Instrument Serif', serif; font-size: clamp(1.7rem, 4vw, 2.2rem);
  margin: 0; letter-spacing: -0.01em; color: var(--text); }
.h2 em { font-style: italic; color: var(--accent); }
.rule { border: 0; border-top: 1px solid var(--line); margin: 3rem 0 2rem 0; }

/* ---------- how-it-works list ---------- */
.steps { margin: 3rem 0 0 0; border-top: 1px solid var(--line); }
.step { display: flex; gap: 1.4rem; padding: 1rem 0; border-bottom: 1px solid var(--line);
  align-items: baseline; }
.step .n { font-family: 'Instrument Serif', serif; font-style: italic; color: var(--mute);
  font-size: .95rem; min-width: 2rem; }
.step .t { color: var(--text); font-weight: 500; min-width: 110px; }
.step .d { color: var(--mute); font-size: .93rem; }

/* ---------- inputs ---------- */
textarea { background: transparent !important; border: 1px solid var(--line) !important;
  border-radius: 4px !important; color: var(--text) !important; line-height: 1.6 !important; }
textarea:focus { border-color: var(--accent) !important; box-shadow: none !important; }
[data-testid="stTextInput"] input, [data-baseweb="select"] > div {
  background: transparent !important; border-color: var(--line) !important;
  border-radius: 4px !important; }

/* ---------- buttons ---------- */
.stButton > button { background: transparent; color: var(--text); border: 1px solid var(--line);
  border-radius: 999px; font-size: .88rem; padding: .35rem 1.1rem; transition: all .15s; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); background: transparent; }
.stButton > button[kind="primary"] { background: var(--accent); color: #0a0b0d; border: 0;
  font-weight: 500; padding: .55rem 1.8rem; }
.stButton > button[kind="primary"]:hover { background: #fff; color: #0a0b0d; }
.stButton > button:disabled { opacity: .35; }

/* ---------- verdict banner ---------- */
.verdict { border: 1px solid var(--v); border-radius: 6px; padding: 1.5rem 1.8rem;
  margin: 2.2rem 0 1.4rem 0;
  box-shadow: 0 0 40px -14px var(--v), inset 0 0 40px -30px var(--v);
  background: color-mix(in srgb, var(--v) 5%, transparent); }
.verdict .tag { font-size: .72rem; letter-spacing: .16em; color: var(--v); font-weight: 500; }
.verdict .msg { font-family: 'Instrument Serif', serif; font-size: clamp(1.8rem, 5vw, 2.6rem);
  line-height: 1.1; margin-top: .3rem; color: var(--text); }
.verdict .when { color: var(--mute); font-size: .8rem; margin-top: .6rem; }

/* ---------- result bits ---------- */
.pill { display: inline-block; margin: 0 8px 8px 0; padding: 4px 13px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--text); font-size: .82rem; }
.reason { color: var(--text); line-height: 1.75; font-size: 1.02rem; max-width: 720px; }
.sim { padding: .9rem 0; border-bottom: 1px solid var(--line); }
.sim .top { font-size: .78rem; color: var(--mute); letter-spacing: .06em; }
.sim .txt { color: var(--text); font-size: .92rem; margin-top: .25rem; opacity: .85; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #0c0d10; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] { padding-top: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .35rem; }
.brand { font-family: 'Instrument Serif', serif; font-size: 1.7rem; color: var(--text); }
.hist { padding-top: .8rem; border-top: 1px solid var(--line); margin-top: .6rem; }
.hist .row1 { display: flex; justify-content: space-between; align-items: center; }
.badge { font-size: .66rem; letter-spacing: .14em; color: var(--v); font-weight: 500; }
.hist .score { color: var(--text); font-size: .85rem; }
.hist .ago { color: var(--mute); font-size: .74rem; }
.hist .prev { color: var(--mute); font-size: .82rem; margin: .25rem 0 .6rem 0; }
[data-testid="stSidebar"] .stButton > button { font-size: .76rem; padding: .1rem .6rem;
  color: var(--mute); white-space: nowrap; width: 100%; }

/* ---------- narrow screens ---------- */
@media (max-width: 700px) {
  .block-container { padding-left: 1.1rem; padding-right: 1.1rem; padding-top: 1.5rem; }
  .step { flex-wrap: wrap; gap: .3rem 1rem; }
  .step .t { min-width: 0; }
  .verdict { padding: 1.1rem 1.2rem; }
}
</style>
"""


def inject():
    """Call once near the top of app.py."""
    st.markdown(CSS, unsafe_allow_html=True)
