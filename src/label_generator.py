"""Teacher label generator: HF Inference API + Chain-of-Thought.

Sends each posting to Llama-3.3-70B-Instruct with a CoT system prompt
and gets back one trust-assessment JSON per posting.

Smoke test:  python -m src.label_generator --dry-run
Full job:    python -m src.label_generator --n 1000

Requires HF_TOKEN in .env. Free tier is rate-limited — full run takes
hours.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.features import extract_all
from src.vector_store import VectorStore

load_dotenv()

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
CONCURRENCY = 3  # free tier is rate-limited; be polite

SYSTEM_PROMPT = """You are a fraud-detection analyst reviewing online job postings.

You receive:
1. The full text of a job posting.
2. A few pre-computed numeric features (urgency words, all-caps ratio, etc.).
3. A neighbor_fraud_rate — the fraction of similar past postings that were fraud.

Think step-by-step before answering, in this order:
1. List suspicious signals you observe in the text.
2. Score each risk category (financial, legitimacy, data_privacy) from 0 to 100, higher = riskier.
3. Decide an action: "avoid", "caution", or "safe".
4. Output ONLY a single JSON object. No markdown fences, no preamble.

Output schema (exact keys):
{
  "trust_score": integer 0-100 (higher = safer),
  "red_flags": list of short strings,
  "risk_breakdown": {
    "financial": integer 0-100,
    "legitimacy": integer 0-100,
    "data_privacy": integer 0-100
  },
  "action": "avoid" or "caution" or "safe",
  "reasoning": 1-3 sentence explanation
}
"""


def build_user_prompt(job_text, features, neighbor_fraud_rate):
    feat_block = "\n".join(f"  {k}: {v:.2f}" for k, v in features.items())
    return (f"Job posting:\n{job_text}\n\n"
            f"Pre-computed numeric features:\n{feat_block}\n"
            f"neighbor_fraud_rate: {neighbor_fraud_rate:.2f}\n\n"
            "Output the JSON now.")


def _parse_json(text):
    """Pull the first balanced {...} JSON object from model text."""
    text = text.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:]
    start = text.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


async def label_one(client, sem, job_text, features, nfr, retries=3):
    """One posting -> one parsed JSON dict (or None on persistent failure)."""
    user = build_user_prompt(job_text, features, nfr)
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.chat_completion(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user}],
                    model=MODEL, max_tokens=512, temperature=0.2,
                )
                return _parse_json(resp.choices[0].message.content)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  [label_one] gave up: {type(e).__name__}: {e}")
                    return None
                await asyncio.sleep(5 * (2 ** attempt))


async def run(input_csv, output_path, n, dry_run):
    import pandas as pd
    from huggingface_hub import AsyncInferenceClient

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. Add it to .env (see .env.example).")

    df = pd.read_csv(input_csv)
    fraud = df[df.fraudulent == 1].sample(
        min(n // 2, (df.fraudulent == 1).sum()), random_state=42)
    real = df[df.fraudulent == 0].sample(n - len(fraud), random_state=42)
    sampled = (pd.concat([fraud, real]).sample(frac=1, random_state=42)
               .reset_index(drop=True))
    if dry_run:
        sampled = sampled.head(5)

    print(f"labelling {len(sampled)} postings via {MODEL} (concurrency={CONCURRENCY})")
    vs = VectorStore()
    client = AsyncInferenceClient(token=token)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def task(row):
        feats = extract_all(row.job_text)
        nfr = vs.neighbor_fraud_rate(row.job_text, k=5)
        label = await label_one(client, sem, row.job_text, feats, nfr)
        if label is None:
            return None
        return {"job_id": int(row.job_id), "fraudulent": int(row.fraudulent),
                "job_text": row.job_text, "neighbor_fraud_rate": nfr,
                **{f"feat_{k}": v for k, v in feats.items()}, **label}

    output = []
    for fut in asyncio.as_completed([task(r) for r in sampled.itertuples()]):
        r = await fut
        if r is not None:
            output.append(r)
            if dry_run or len(output) % 25 == 0:
                print(f"  {len(output)} / {len(sampled)} done")

    if dry_run:
        print(json.dumps(output, indent=2))
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in output:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(output)} labels to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/train.csv")
    p.add_argument("--output", default="data/labeled/train.jsonl")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true",
                   help="label 5 rows and print, do not write a file")
    args = p.parse_args()
    asyncio.run(run(args.input, args.output, args.n, args.dry_run))


if __name__ == "__main__":
    main()
