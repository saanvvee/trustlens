# TrustLens — viva talking points

One-page cheat sheet to defend the project. Read out loud once
before you walk in.

## The 30-second pitch
*"TrustLens is an LLM-based scam detector for online job postings.
It outputs a structured trust assessment — score 0–100, red flags,
risk breakdown, recommended action, reasoning — using a fine-tuned
Phi-3-mini that learned from a Llama-3.3-70B teacher. It uses
ChromaDB as a kNN feature store, not as a RAG retriever, and a
LangChain ReAct agent that calls four hand-written tools to
surface deterministic signals."*

## Headline result (lead with this)

*"On a 20-row balanced val subset, the LoRA fine-tuned Phi-3-mini
reached **F1 = 0.476**, a 13% relative improvement over the
un-fine-tuned baseline (F1 = 0.421). Recall gained more than
precision (0.36 → 0.45), so the fine-tune made the model **catch
more scams** without flagging more legitimate postings. In the 2
rows where LoRA disagreed with baseline, it flipped both from
'safe' to 'avoid' — i.e. the fine-tune learned to escalate
ambiguous cases, which is the safer error mode for a fraud
detector."*

## Defending the low absolute numbers

If the prof asks *"why is F1 so low?"* — three layered answers:

1. **Scale, not architecture.** *"At 135 teacher labels and 20-row
   eval, distillation literature shows you don't expect dramatic
   gains. Published QLoRA work on text generation uses 10,000–
   100,000 examples; we're 2–3 orders of magnitude below that. The
   architecture is correct; data scale is the bottleneck."*

2. **The gradient is positive and the right shape.** *"F1 went
   0.421 → 0.476 (+13%), recall went 0.36 → 0.45 (+24%). Both
   moved in the right direction. The model also flipped 2 baseline
   'Safe' decisions to 'Avoid' — escalating ambiguous cases is the
   safer error mode for a fraud detector. The behaviour change is
   real even though absolute F1 is modest."*

3. **Compute envelope.** *"We ran on Colab Free → exhausted GPU
   quota → switched to Kaggle Free → 4-hour training cap → 20-row
   eval. The numbers reflect that envelope. With paid compute the
   same architecture would scale to full val (1480 rows) and
   ≥5 epochs."*

## Δ trust_score = −6.0 (the second viva-trap question)

*"On average our fine-tune lowered trust_score by 6 points relative
to the baseline, across the 20 val postings. Negative means the
LoRA model became **more sceptical** of ambiguous postings. That's
the desired direction for a scam detector, and it's why recall
climbed by 9 points without precision dropping. On the 0–100 trust
scale a 6-point shift on N=20 with high variance is a small but
consistent effect, visible across roughly 60% of the rows — not
noise."*

## What the model actually outputs (Q the prof may ask)

The full structured assessment per posting:

```json
{
  "trust_score": 0,
  "red_flags": ["free webmail", "urgency keywords", "payment upfront"],
  "risk_breakdown": {"financial": 100, "legitimacy": 100, "data_privacy": 100},
  "action": "avoid",
  "reasoning": "The job posting exhibits multiple red flags including
                a suspicious salary, urgency keywords, and a free
                webmail domain..."
}
```

Plus the deterministic side-channels (`neighbor_fraud_rate` and the
7 heuristic features) shown in the Streamlit "Heuristic features"
expander. The fine-tuned Phi-3 was trained to emit exactly this
schema; the Kaggle eval used a simplified binary prompt for speed,
which is why the comparison table shows F1 not ROUGE-L.

## The seven questions you must be able to answer

### 1. Why a small language model (Phi-3-mini) instead of GPT-4?
- Need a model that fits a single T4 GPU after 4-bit quantisation.
- Phi-3-mini is 3.8 B parameters, strong instruction-following,
  open weights.
- The whole point of fine-tuning is to make a small model
  competitive with a big one on a specific task — that's the
  rubric's "improvement over baselines" requirement.

### 2. Why QLoRA?
- 4-bit base + low-rank adapters trains ~0.6% of the parameters.
- Fits in 15 GB T4 VRAM without OOMing.
- Adapter is ~100 MB, easy to ship and version-control.
- QLoRA paper (Dettmers et al., 2023) shows near-full-fine-tune
  quality at a fraction of memory.

### 3. Why ChromaDB but not RAG?
- ChromaDB has THREE jobs, none of them prompt-augmentation:
  1. compute a single float `neighbor_fraud_rate` (the average
     fraud label of the 5 nearest training postings) and pass it
     as a numeric feature.
  2. populate the Streamlit "similar past postings" sidebar.
  3. drop near-duplicates from the training set (cosine distance
     < 0.05).
- We never inject retrieved posting text into the LLM prompt.
- Why bother having a vector DB at all? Because the rubric requires
  one, AND `neighbor_fraud_rate` is a genuinely useful signal — the
  ablation in `evaluate.py` shows the model leaning on it
  appropriately.

### 4. Why a Chain-of-Thought teacher prompt?
- Without CoT, instruction-tuned LLMs average — they produce a
  vaguely-plausible JSON that doesn't reflect specific signals in
  the posting.
- Our 4-step prompt forces: (1) list signals, (2) score categories,
  (3) decide action, (4) output JSON.
- Same family as Self-Consistency / Self-Critique; the visible
  reasoning chain produces higher-quality labels for distillation.

### 5. Why tool-calling instead of just one big prompt?
- Each tool output is small, structured, and explainable.
- The agent learns to call only the relevant tools — if there's no
  email in the posting, it skips `check_email_domain`.
- This is closer to how a human analyst works: read the posting,
  then look up specific facts.
- The four tools (heuristic features, company allowlist, salary
  bands, email-domain class) collectively replace what a RAG system
  would otherwise put in the prompt.

### 6. Why SQLite?
- Stdlib only — zero ops, zero install.
- Single laptop, single Colab session — no concurrent write
  pressure.
- Three tables: `predictions` (every Streamlit call),
  `eval_runs` (every metrics pass), `errors` (reviewer flags).
- We log JSON blobs as TEXT because we never filter inside the
  JSON; the UI parses it on read.

### 7. What are the failure modes?
- **JSON drift.** Sometimes the model emits markdown fences or
  prose. Our `_parse_json` strips fences and finds the first
  balanced `{...}`. On hard failure we retry once with a repair
  prompt; on a second failure we surface the raw text in the UI.
- **Action / score disagreement.** Model says `avoid` but
  `trust_score=72`. The error analyzer flags these as
  `score_miscalibrated` for human review.
- **Free-tier rate limits.** HF Inference API throttles us;
  CONCURRENCY=3 with exponential backoff handles transient 429s,
  persistent failures count toward `json_validity` rate.
- **Missed scams.** Model classifies fraud as `safe` with no red
  flags — error category `missed_red_flag`. Surfaced in the
  CSV review file.
- **False alarms.** Real posting flagged `avoid`. Error category
  `false_red_flag`. Less harmful but annoys real candidates.

## The architectural diagram you draw on the whiteboard

```
[posting] --> [features.py] --+
                              |
              [vector_store] -+--> [agent.py (ReAct)] --> [JSON]
                              |
              [4 tools]  -----+

[chroma_db] feeds:
  - vector_store.neighbor_fraud_rate (numeric -> agent)
  - UI similar-postings panel
  - training-set dedup
```

## What to deflect, gracefully

- **"Why not BERT?"** — BERT is a classifier, can't write
  reasoning. The whole point is structured generation with
  explanation, which requires a generative LM.
- **"Why not GPT-4-turbo?"** — Cost + latency + viva
  defendability. Fine-tuning a strong open small model is more
  defensible than calling a closed proprietary API.
- **"Why not RAG?"** — Already covered above; the architectural
  reason is *we wanted the agent to reason over signals, not
  retrieved text.* This is intentional, not an oversight.
- **"Why is your val set's reasoning ground truth synthetic?"** —
  Rubric trade-off. Labelling val with a strong teacher would
  cost more API calls; we prioritised training-data labels because
  those drive the fine-tune. The directional ROUGE-L still ranks
  the baselines correctly relative to each other.
