"""All the custom CSS for the app, in one place.

Direction: light, airy and calm. Pale blue-white page, white cards with big
radii and soft shadows, one desaturated blue accent, pastel verdict colours.
A heavily blurred cloud photo (assets/clouds.jpg, public domain) sits fixed
behind everything at very low opacity and fades out toward the bottom so
text stays readable.

The verdict colours are passed in per-element from app.py through the CSS
variables --v (dot / bar), --vt (soft tint) and --vd (dark text version).
"""
import base64
import os

import streamlit as st

CLOUDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "clouds.jpg")


@st.cache_resource
def _clouds_css():
    """The photo inlined as a data URI so no static-file serving is needed.
    If the file is missing the page just has no clouds — nothing crashes."""
    try:
        with open(CLOUDS_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"url('data:image/jpeg;base64,{b64}')"
    except OSError:
        return "none"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@1&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg: #F4F7FA;
  --ink: #141618;
  --body: #5A6672;
  --accent: #7FA3C4;
  --soft: rgba(20, 22, 24, 0.07);
  --shadow: 0 10px 40px rgba(70, 100, 130, 0.10);
  --shadow-sm: 0 2px 14px rgba(70, 100, 130, 0.10);
}

/* ---------- base ---------- */
/* streamlit sets its own font on nearly every element, so override them all
   (except icons, code, the json viewer and the one italic serif word) */
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre):not(em):not([data-testid="stJson"] *) {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp .display em { font-family: 'Instrument Serif', serif !important; }
.stApp { background: var(--bg); color: var(--body); }
.stMarkdown p, .stMarkdown li, label, [data-testid="stCaptionContainer"] {
  color: var(--body); font-size: 1.05rem; line-height: 1.6;
}

/* clouds: a clearly visible sky band behind the hero, fixed to the top of
   the window and faded out with a mask so the lower page stays clean */
.stApp::before {
  content: ""; position: fixed; top: -20px; left: -20px; right: -20px; height: 80vh;
  z-index: 0; pointer-events: none;
  background: __CLOUDS__ center / cover no-repeat;
  filter: blur(5px); opacity: 0.6;
  -webkit-mask-image: linear-gradient(to bottom, #000 0%, #000 30%, transparent 100%);
  mask-image: linear-gradient(to bottom, #000 0%, #000 30%, transparent 100%);
}
[data-testid="stAppViewContainer"] { position: relative; z-index: 1; background: transparent; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"] { background: transparent; }
[data-testid="stSidebar"] { z-index: 2; }

/* hide streamlit chrome. The header stays (transparent) because the
   sidebar toggle lives in it on narrow screens. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.5rem; padding-bottom: 6rem; max-width: 980px; }

/* ---------- type ---------- */
.eyebrow { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase;
  color: #3B4651; font-weight: 600; margin: 0 0 1rem 0; opacity: .85; }
.display { font-weight: 600; font-size: clamp(2.7rem, 7.5vw, 5rem); line-height: 1.02;
  letter-spacing: -0.03em; word-spacing: .06em; color: var(--ink); margin: 0; }
.display em { font-style: italic; font-weight: 400;
  font-size: 1.1em; letter-spacing: -0.01em; }
.stMarkdown .lede { color: #3B4651; font-size: 1.12rem; line-height: 1.65; max-width: 580px;
  margin: 2.6rem 0 0 0; }

/* ---------- how-it-works list ---------- */
.steps { margin: 3.2rem 0 3.5rem 0; }
.step { display: flex; gap: 1.5rem; padding: 1.1rem 0; align-items: baseline;
  border-bottom: 1px solid var(--soft); }
.step:first-child { border-top: 1px solid var(--soft); }
.step .n { color: #3B4651; font-size: .85rem; font-weight: 500; min-width: 2rem; }
.step .t { color: var(--ink); font-weight: 600; min-width: 120px; }
.step .d { color: #3B4651; font-size: .98rem; }

/* ---------- inputs ---------- */
[data-baseweb="textarea"] { background: #fff !important; border: 0 !important;
  border-radius: 20px !important; box-shadow: var(--shadow-sm); transition: box-shadow .15s; }
[data-baseweb="textarea"]:focus-within { box-shadow: 0 0 0 3px rgba(127,163,196,.35), var(--shadow-sm); }
textarea { background: transparent !important; color: var(--ink) !important;
  line-height: 1.6 !important; padding: 1.1rem 1.3rem !important; font-size: 1rem !important; }
textarea::placeholder { color: #8B97A3 !important; }
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="input"] { background: #F1F5F9 !important;
  border: 0 !important; border-radius: 12px !important; }
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within { box-shadow: 0 0 0 3px rgba(127,163,196,.35); }
input { color: var(--ink) !important; background: transparent !important; }

/* ---------- buttons ---------- */
.stButton > button { background: #fff; color: var(--ink); border: 0; border-radius: 999px;
  box-shadow: var(--shadow-sm); font-weight: 500; font-size: .92rem; padding: .5rem 1.4rem;
  transition: transform .15s, box-shadow .15s; white-space: nowrap; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: var(--shadow);
  color: var(--ink); background: #fff; }
.stButton > button[kind="primary"] { background: var(--ink); color: #fff;
  padding: .7rem 2rem; font-weight: 600; }
.stButton > button[kind="primary"]:hover { background: #2a2f33; color: #fff; }
.stButton > button:disabled { opacity: .45; box-shadow: none; transform: none; }

/* ---------- status + expander ---------- */
[data-testid="stStatus"] details, [data-testid="stExpander"] details {
  background: #fff; border: 0 !important; border-radius: 16px; box-shadow: var(--shadow-sm); }
[data-testid="stStatus"] summary p, [data-testid="stExpander"] summary p { color: var(--ink); }

/* ---------- result card ---------- */
.st-key-resultcard { background: #fff; border-radius: 24px; box-shadow: var(--shadow);
  padding: 2.2rem 2.4rem; margin-top: 2.5rem; }
.verdict { background: var(--vt); border-radius: 16px; padding: 1.5rem 1.8rem;
  margin-bottom: 2.2rem; }
.verdict .tag { font-size: .74rem; letter-spacing: .14em; color: var(--vd); font-weight: 700; }
.verdict .msg { font-size: clamp(1.6rem, 4.5vw, 2.3rem); font-weight: 600; line-height: 1.15;
  letter-spacing: -0.02em; color: var(--ink); margin-top: .35rem; }
.verdict .when { color: var(--body); font-size: .85rem; margin-top: .6rem; }
.pill { display: inline-block; margin: 0 8px 8px 0; padding: 6px 14px; border-radius: 999px;
  background: rgba(217,130,130,.14); color: #8F4545; font-size: .86rem; font-weight: 500; }
.reason { color: var(--ink); line-height: 1.7; font-size: 1.05rem; max-width: 700px; opacity: .88; }
.sim { padding: 1rem 0; border-bottom: 1px solid var(--soft); }
.sim:last-child { border-bottom: 0; }
.sim .top { font-size: .74rem; letter-spacing: .1em; color: var(--body); font-weight: 600; }
.sim .txt { color: var(--ink); opacity: .8; font-size: .95rem; margin-top: .3rem; line-height: 1.55; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #fff; border: 0; box-shadow: 6px 0 40px rgba(70,100,130,.10); }
[data-testid="stSidebarContent"] { padding-top: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .6rem; }
.brand { font-weight: 700; font-size: 1.5rem; letter-spacing: -0.03em; color: var(--ink); }
[class*="st-key-hist-"] { background: #fff; border-radius: 16px; padding: .95rem 1rem .5rem 1rem;
  box-shadow: 0 0 0 1px rgba(20,22,24,.04), var(--shadow-sm); gap: .25rem !important; }
.hist .row1 { display: flex; align-items: center; gap: .55rem; }
.hist .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--v); }
.hist .score { color: var(--ink); font-weight: 600; font-size: .9rem; }
.hist .ago { color: #6F7C89; font-size: .76rem; margin-left: auto; }
.hist .prev { color: var(--body); font-size: .84rem; line-height: 1.45; margin: .35rem 0 .6rem 0; }
/* quiet text buttons in the sidebar: colour only on hover */
[data-testid="stSidebar"] .stButton > button { background: transparent; box-shadow: none;
  color: #6F7C89; padding: .1rem .2rem; font-size: .8rem; font-weight: 500;
  white-space: nowrap; width: auto; justify-content: flex-start; }
[data-testid="stSidebar"] .stButton > button:hover { color: var(--accent); background: transparent;
  transform: none; box-shadow: none; }

/* ---------- narrow screens ---------- */
@media (max-width: 700px) {
  .block-container { padding-left: 1.1rem; padding-right: 1.1rem; padding-top: 2.2rem; }
  .step { flex-wrap: wrap; gap: .3rem 1rem; }
  .step .t { min-width: 0; }
  .st-key-resultcard { padding: 1.3rem 1.2rem; border-radius: 20px; }
  .verdict { padding: 1.1rem 1.2rem; }
}
</style>
"""


def inject():
    """Call once near the top of app.py."""
    st.markdown(CSS.replace("__CLOUDS__", _clouds_css()), unsafe_allow_html=True)
