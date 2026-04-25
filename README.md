# TrustLens

An LLM-powered tool that evaluates online job and freelance postings for
scam risk and produces a structured trust assessment.

For each posting it returns:
- `trust_score` — 0 to 100
- `red_flags` — list of suspicious signals
- `risk_breakdown` — `financial`, `legitimacy`, `data_privacy` sub-scores
- `action` — `avoid`, `caution`, or `safe`
- `reasoning` — short natural-language explanation

## Storage architecture

> **ChromaDB is used for kNN risk features, UI similarity panels, and
> training-data deduplication. It is *not* used for prompt augmentation.
> This is intentionally not RAG.**

- **ChromaDB** holds embeddings of historical postings. We use it to:
  1. compute a `neighbor_fraud_rate` numeric feature from the labels of
     the K-nearest training postings,
  2. show the 3 most-similar past postings in the Streamlit UI, and
  3. drop near-duplicates from the training set before fine-tuning.
- **SQLite** logs every prediction the Streamlit app makes and every
  evaluation run.

## Stack

- Base model: `microsoft/Phi-3-mini-4k-instruct` quantised to 4-bit.
- Fine-tuning: QLoRA via vanilla `transformers` + `peft` (no Unsloth)
  on Google Colab Free, single T4 GPU.
- Teacher labels: GPT-4o-mini with a Chain-of-Thought prompt.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` on CPU.
- Orchestration: LangChain ReAct agent with 4 tools.
- Frontend: Streamlit.

## Layout

```
trustlens/
├── data/{raw,processed,labeled}/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_label_generation.ipynb
│   └── 03_fine_tune.ipynb        # runs on Colab
├── src/
│   ├── features.py
│   ├── vector_store.py
│   ├── db.py
│   ├── label_generator.py
│   ├── baselines.py
│   ├── tools.py
│   ├── agent.py
│   ├── pipeline.py
│   ├── evaluate.py
│   └── error_analyzer.py
├── tests/
├── docs/                          # study sheets per file
├── chroma_db/                     # gitignored
├── models/                        # gitignored
├── app.py                         # Streamlit
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in OPENAI_API_KEY
```

The fine-tuning step runs on Colab Free; everything else runs locally on CPU.

A full "Why each choice", "How to run", and rubric-mapping section will be
added in the final polish step (STEP 18).
