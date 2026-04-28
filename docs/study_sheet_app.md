# Study sheet — `app.py`

The Streamlit frontend the prof clicks through during the demo.
Renders the structured output of `pipeline.analyze_job` as a gauge,
risk-breakdown bar chart, action badge, red-flag chips, and
expandable reasoning + features panel. Sidebar shows recent
analyses (from SQLite) and similar past postings (from ChromaDB).

## What each part does

### Setup at the top
- `set_page_config(layout="wide")` — wider page for the two-column
  result layout.
- `ACTION_COLORS = {avoid: red, caution: amber, safe: green}` — the
  one piece of "design language" we have. Used for the action badge.

### `_gauge(score)`
Plotly indicator. The arc has 3 colour bands: 0–30 red, 30–70
amber, 70–100 green. The needle sits at the predicted trust_score.
Sized 260 px tall to fit alongside the risk bars.

### `_risk_bars(rb)`
Horizontal bar chart of the three sub-scores from `risk_breakdown`.
Bar colours flip from green to amber to red at 30 and 60 — same
thresholds as the gauge for visual consistency.

### `@st.cache_resource def _vs()`
Streamlit reruns the script on every widget interaction; without
caching, every keystroke would reload ChromaDB. `cache_resource`
caches the VectorStore instance for the whole user session.

### Sidebar
- "Recent analyses" — `get_recent_predictions(limit=10)`. Each row
  is a small bordered container showing action + score + the first
  120 chars of the posting.
- "Similar past postings" — populated *after* an Analyze click via
  `_vs().query(posting, k=3)`. Each result shows fraud emoji
  (🚨/✅), cosine distance, and a 160-char preview. The placeholder
  exists at first render and gets filled by the click handler.

### Main area
- One big text area for the posting. The Analyze button is disabled
  until the user has typed at least 50 characters.
- On click: spinner, call `analyze_job`, render or surface error.
- Result layout: action banner (large, coloured) → two columns
  (gauge + risk bars) → red-flag chips → reasoning expander →
  raw-features expander.

### Red-flag rendering
Each chip is a pink rounded `<span>` rendered with
`unsafe_allow_html=True`. We pass HTML directly because Streamlit's
native components don't have a "chip" widget and a multi-column
button row would look worse.

## Sample viva Q&A

**Q: Why Streamlit and not Flask + a custom React frontend?**
A: One file vs a frontend + a backend. Streamlit's reactive model
fits the demo: text-area in, structured object out, plotly charts,
sidebar with state from SQLite. A "real" production system would
likely be FastAPI + a React UI, but for a course demo the trade-off
goes the other way — fewer moving parts to break, and the prof can
read the entire UI implementation in one file.

**Q: Why the analyze button is disabled at <50 characters?**
A: The cheapest way to stop the user from clicking through with
empty input. The label_generator at STEP 8 saw postings averaging
1000+ characters; the agent's prompt assumes meaningful text. 50
chars is a polite floor — enough to carry signal, low enough not to
block legitimately short test inputs.

**Q: How does this UI behave differently when the backend is the
fine-tuned Phi-3 vs the HF API?**
A: It doesn't. `pipeline.analyze_job` returns the same dict
whichever backend `agent.py` is using. The UI doesn't know — that's
the whole point of putting the backend choice inside the agent
factory function. For the local demo, no GPU available, the agent
quietly uses Llama-3.3-70B via HF API; for the proper STEP 14
evaluation on Colab, the agent loads Phi-3 + the LoRA adapter. Same
JSON shape either way.
