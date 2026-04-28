# TrustLens

LLM-powered tool that evaluates online job/freelance postings for
scam risk and produces a structured trust assessment.

For each posting it returns:
- `trust_score` — integer 0 to 100, higher is safer
- `red_flags` — short list of suspicious signals
- `risk_breakdown` — sub-scores for `financial`, `legitimacy`, `data_privacy`
- `action` — one of `avoid`, `caution`, `safe`
- `reasoning` — 1–3 sentence natural-language explanation

## Problem

Online job boards leak scams that look like legitimate openings.
Common patterns: unrealistic pay, urgent hiring, payment-upfront
requests, contact via free webmail, vague company descriptions. A
binary "is fraud" classifier doesn't help a candidate — they want
to know *why* a posting looks risky. TrustLens produces a
structured assessment with reasoning so the candidate can decide.

## Architecture overview

```
                  ┌─────────────────────────────────────┐
                  │  Streamlit UI (app.py)              │
                  └────────────────┬────────────────────┘
                                   │ analyze_job(text)
                  ┌────────────────▼────────────────────┐
                  │  src/pipeline.py                    │
                  │   1. extract_all (heuristic feats)  │
                  │   2. neighbor_fraud_rate (ChromaDB) │
                  │   3. agent.invoke (ReAct)           │
                  │   4. parse JSON, retry on fail      │
                  │   5. log_prediction (SQLite)        │
                  └────────────────┬────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              │                    │                     │
       ┌──────▼─────┐      ┌───────▼──────┐      ┌───────▼──────┐
       │ src/tools  │      │ Fine-tuned    │      │ ChromaDB     │
       │  (4 tools) │      │ Phi-3 + LoRA  │      │ (8866 docs)  │
       └────────────┘      │ — or —        │      └──────────────┘
                           │ Llama-3.3-70B │
                           │  via HF API   │
                           └───────────────┘
```

**The agent never sees retrieved text from past postings.** ChromaDB
contributes a single number (`neighbor_fraud_rate`), powers the UI
similarity panel, and runs training-set deduplication. This is
intentionally *not* RAG.

## Storage architecture

> **ChromaDB is used for kNN risk features, UI similarity panels, and
> training-data deduplication. It is *not* used for prompt
> augmentation.**

- **ChromaDB** holds embeddings of historical postings:
  1. computes a `neighbor_fraud_rate` numeric feature from the
     labels of the K-nearest training postings,
  2. shows the 3 most-similar past postings in the Streamlit UI, and
  3. drops near-duplicates (cosine distance < 0.05) from the
     training set before fine-tuning.
- **SQLite** logs every prediction the Streamlit app makes, every
  evaluation run, and every reviewer-flagged error.

## Stack

- **Base model:** `microsoft/Phi-3-mini-4k-instruct` quantised to
  4-bit (NF4) for fine-tuning.
- **Fine-tuning:** QLoRA via vanilla HuggingFace `transformers` +
  `peft` on Google Colab Free T4. No Unsloth.
- **Teacher labels:** `meta-llama/Llama-3.3-70B-Instruct` via the
  HuggingFace Inference API, with a 4-step Chain-of-Thought prompt.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` on CPU,
  384-dim, cosine similarity.
- **Vector DB:** ChromaDB PersistentClient, collection `trustlens`.
- **Relational DB:** SQLite via stdlib `sqlite3`. No ORM.
- **Orchestration:** LangChain ReAct agent over 4 hand-written tools.
- **Frontend:** Streamlit + Plotly.
- **Evaluation:** ROUGE-L on reasoning, F1 on action, MAE +
  Pearson r on trust_score, JSON-validity rate.

## Why each choice

**Why a small language model (Phi-3-mini) over GPT-4?**
We need a model that can be fine-tuned and served without paid
infrastructure. Phi-3-mini is 3.8 B parameters, fits in 4-bit on a
single T4, and has strong instruction-following. The student
imitates a 70B teacher; the rubric explicitly rewards
"improvement-over-baselines," which we get from the
teacher→student distillation.

**Why QLoRA?**
4-bit base + low-rank adapters means we train ~0.6% of the
parameters on a 15 GB T4 GPU without OOMing. The QLoRA paper (Dettmers
et al., 2023) shows this achieves near-full-fine-tune quality at a
fraction of the memory cost. The full project budget is one Colab
Free session.

**Why ChromaDB without RAG?**
Two reasons. (i) The course rubric distinguishes "vector DB" from
"RAG" — they want a vector DB *somewhere*, not specifically as
prompt augmentation. (ii) Tool-calling + numeric features is
cleaner to defend in viva than RAG: every signal that reaches the
LLM is either the posting text, a numeric feature, or a structured
tool output. No retrieved text, no prompt-injection surface area,
deterministic input shape.

**Why a Chain-of-Thought teacher prompt?**
Without CoT the teacher LLM averages — it produces a vaguely
plausible JSON that doesn't reflect the specific posting. The
4-step prompt ("list signals → score categories → decide action →
output JSON") forces grounded reasoning and gives us higher-quality
labels for the student to imitate.

**Why tool-calling at all?**
The agent calls tools to surface deterministic, explainable signals
mid-thought (heuristic features, company allowlist, salary bands,
email-domain class). Each tool output is a small structured string,
not retrieved text. This pushes the model toward the same kind of
analyst process a human would follow: read the posting first, then
look up specifics.

**Why SQLite?**
Stdlib only. Zero ops, zero install, zero credentials. The whole
project runs on one laptop and one Colab session — we never need
concurrent write throughput. PostgreSQL would force docker-compose
for no benefit.

## Layout

```
trustlens/
├── data/
│   ├── raw/                          # Kaggle CSV (gitignored)
│   ├── processed/                    # train/val/test splits (gitignored)
│   └── labeled/                      # teacher labels + baseline predictions
├── notebooks/
│   ├── 01_eda.ipynb                  # local: EDA + stratified split
│   ├── 02_label_generation.ipynb     # placeholder; logic lives in src/
│   ├── 03_fine_tune.ipynb            # Colab: QLoRA fine-tuning
│   └── trustlens_colab.ipynb         # Colab: end-to-end demo
├── src/
│   ├── features.py                   # 7 heuristic signals
│   ├── vector_store.py               # ChromaDB wrapper
│   ├── db.py                         # SQLite logging
│   ├── label_generator.py            # CoT teacher prompt + async HF client
│   ├── baselines.py                  # B1/B2/B3/B4
│   ├── tools.py                      # 4 LangChain @tool functions
│   ├── agent.py                      # ReAct AgentExecutor builder
│   ├── pipeline.py                   # analyze_job(text) — end-to-end
│   ├── evaluate.py                   # metrics + comparison table
│   └── error_analyzer.py             # CLI for qualitative review
├── scripts/
│   └── build_chroma_index.py         # one-shot index build from train.csv
├── tests/                            # 29 pytest cases
├── docs/                             # study sheets, one per src/ file
├── results/                          # comparison_table.md + comparison.png
├── chroma_db/                        # gitignored
├── models/                           # gitignored (LoRA adapter lives in Drive)
├── app.py                            # Streamlit
├── requirements.txt
├── .env.example                      # HF_TOKEN= (placeholder)
├── .gitignore
└── README.md
```

## Setup (local Mac demo)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # bitsandbytes will fail on macOS — that's fine
cp .env.example .env                   # then paste your HF token after HF_TOKEN=
```

Get a free HuggingFace token at
[huggingface.co](https://huggingface.co) → Settings → Access Tokens.
Request access to `meta-llama/Llama-3.3-70B-Instruct` (one-time
license click).

## How to run

### 1. EDA + splits (local, ~1 min)
```bash
jupyter notebook notebooks/01_eda.ipynb   # then Run All
```
Produces `data/processed/{train,val,test}.csv`.

### 2. Build the ChromaDB index (local, ~3 min)
```bash
python -m scripts.build_chroma_index
```
Persists to `./chroma_db/` (190 MB, gitignored).

### 3. Generate teacher labels (local, hours on free tier)
```bash
python -m src.label_generator --dry-run     # smoke test, 5 rows
python -m src.label_generator --n 1000      # full job
```
Writes `data/labeled/train.jsonl`.

### 4. Fine-tune Phi-3 (Colab, ~30–60 min on T4)
Upload `notebooks/03_fine_tune.ipynb` to Colab, runtime → T4, run
all cells. The LoRA adapter saves to your Drive at
`MyDrive/trustlens/phi3-trustlens-lora/`.

### 5. Run baselines (Colab)
Open a second Colab tab on `notebooks/trustlens_colab.ipynb`,
section 6. Runs B4 DistilBERT (~5 min on T4) and B3 Llama
few-shot. B1/B2 (Phi-3 zero-/few-shot) commented-out — uncomment
to run.

### 6. Evaluate (Colab)
Section 8 of `trustlens_colab.ipynb` runs `python -m src.evaluate`,
which writes `results/comparison_table.md` and `results/comparison.png`.

### 7. Streamlit demo (local Mac)
```bash
streamlit run app.py
```
The agent uses Llama-3.3-70B via HF API on Mac (no GPU needed).

## Dependencies

Pinned in `requirements.txt`. Highlights:

- transformers 4.45.2, peft 0.13.2, bitsandbytes 0.44.1
  (CUDA-only), accelerate 1.0.1, datasets 3.0.2, torch 2.4.1
- langchain 0.3.7, langchain-community 0.3.5, langchain-huggingface
- sentence-transformers 3.2.1, chromadb 0.5.15
- streamlit 1.39.0, plotly 5.24.1
- rouge-score 0.1.2, pytest 8.3.3, python-dotenv 1.0.1

## Compute requirements

- **Local Mac (any architecture, ≥ 8 GB RAM):** runs the splits,
  ChromaDB build, label generation (HF API), and Streamlit demo.
- **Colab Free T4 (15 GB VRAM):** runs the QLoRA fine-tune and the
  Phi-3 baselines (B1/B2/B4).

## Rubric mapping

| Rubric item | Where it's satisfied |
|---|---|
| Application-specific dataset, preprocessed, stratified train/val/test split | `notebooks/01_eda.ipynb` (STEP 3) |
| PEFT fine-tuning (QLoRA on Phi-3-mini) with written justification | `notebooks/03_fine_tune.ipynb` + the "Why each choice" section above |
| Baseline comparison: zero-shot LLM, few-shot LLM, non-LLM classifier | `src/baselines.py` — B1 (Phi-3 zero-shot), B2 (Phi-3 few-shot), B3 (Llama-3.3-70B few-shot), B4 (DistilBERT) |
| Dual storage: ChromaDB (vector) + SQLite (relational) | `src/vector_store.py` + `src/db.py` |
| Quantitative eval: ROUGE-L, F1, MAE, JSON-validity | `src/evaluate.py` → `results/comparison_table.md` |
| Qualitative + error analysis | `src/error_analyzer.py` → `results/errors_for_review.csv` |
| Improvement over baselines + real-world applicability | `results/comparison.png` + Streamlit demo (`app.py`) |

## Limitations and future work

- The fine-tuned Phi-3 path requires CUDA + bitsandbytes; the local
  Streamlit demo therefore uses the HF API as a fallback. The
  metrics in `results/comparison_table.md` come from the proper
  Phi-3 path on Colab.
- Val/test labels are the original Kaggle binary `fraudulent` —
  per-row reasoning ground truth would let us compute absolute
  ROUGE-L instead of the directional version we use.
- The company-allowlist tool is a 30-name hardcoded list; a real
  production system would query Crunchbase or SEC EDGAR.
- Free-tier HF inference rate-limits force CONCURRENCY=3 and slow
  the labelling job. A paid tier would shrink that to minutes.

## Tests

```bash
.venv/bin/python -m pytest -q
```
29 cases covering features, db, and label_generator. The agent /
pipeline / Streamlit pieces are exercised end-to-end by the demo
notebook rather than unit tests.
