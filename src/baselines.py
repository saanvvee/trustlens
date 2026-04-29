"""Four baselines on the val set, each writing data/labeled/baseline_<name>.jsonl.

B1 phi3_zeroshot   — Phi-3-mini, schema-only prompt, no examples
B2 phi3_fewshot    — Phi-3-mini, 3 in-context examples from teacher labels
B3 llama_fewshot   — Llama-3.3-70B-Instruct via HF Inference API, 3 examples
B4 distilbert      — DistilBERT trained on binary fraud label only

CLI:
    python -m src.baselines --which b4    # one
    python -m src.baselines --which all   # everything (Colab recommended for B1/B2)

B2/B3 require data/labeled/train.jsonl (run src.label_generator first).
"""
import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.features import extract_all
from src.label_generator import (CONCURRENCY, SYSTEM_PROMPT, _parse_json,
                                 build_user_prompt)
from src.vector_store import VectorStore

load_dotenv()

VAL_CSV = "data/processed/val.csv"
TRAIN_CSV = "data/processed/train.csv"
TRAIN_JSONL = "data/labeled/train.jsonl"
OUT_DIR = Path("data/labeled")

PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct"
LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

LABEL_KEYS = ["trust_score", "red_flags", "risk_breakdown", "action", "reasoning"]


def _load_examples(n=3):
    if not Path(TRAIN_JSONL).exists():
        raise SystemExit(f"missing {TRAIN_JSONL} — run src.label_generator first")
    rows = [json.loads(l) for l in open(TRAIN_JSONL)]
    random.seed(42)
    return random.sample(rows, min(n, len(rows)))


def _build_messages(job_text, features, nfr, examples=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples or []:
        ex_feats = {k.replace("feat_", ""): v
                    for k, v in ex.items() if k.startswith("feat_")}
        msgs.append({"role": "user",
                     "content": build_user_prompt(ex["job_text"], ex_feats,
                                                  ex["neighbor_fraud_rate"])})
        msgs.append({"role": "assistant",
                     "content": json.dumps({k: ex[k] for k in LABEL_KEYS})})
    msgs.append({"role": "user",
                 "content": build_user_prompt(job_text, features, nfr)})
    return msgs


def baseline_phi3(name, fewshot=False, val_csv=VAL_CSV):
    """B1 zero-shot or B2 few-shot on Phi-3-mini. Loads model in fp16."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(PHI3_MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        PHI3_MODEL, torch_dtype=torch.float16, device_map="auto")
    examples = _load_examples(3) if fewshot else None

    df = pd.read_csv(val_csv)
    vs = VectorStore()
    out = []
    for row in df.itertuples():
        feats = extract_all(row.job_text)
        nfr = vs.neighbor_fraud_rate(row.job_text, k=5)
        msgs = _build_messages(row.job_text, feats, nfr, examples)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3000).to(mdl.device)
        gen = mdl.generate(**ids, max_new_tokens=512, do_sample=False)
        text = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = _parse_json(text) or {}
        out.append({"job_id": int(row.job_id), "fraudulent": int(row.fraudulent), **parsed})
        if len(out) % 50 == 0:
            print(f"  phi3 {name}: {len(out)} / {len(df)}")
    _write(name, out)


async def baseline_llama_fewshot(val_csv=VAL_CSV):
    """B3: Llama-3.3-70B via HF Inference API with 3 in-context examples."""
    from huggingface_hub import AsyncInferenceClient
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set; add to .env")
    examples = _load_examples(3)
    df = pd.read_csv(val_csv)
    vs = VectorStore()
    # provider="hf-inference" forces HF's legacy direct route instead of
    # auto-routing to Groq, which requires a token with inference-provider
    # access that free Read tokens don't have.
    client = AsyncInferenceClient(token=token, provider="hf-inference")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def task(row):
        feats = extract_all(row.job_text)
        nfr = vs.neighbor_fraud_rate(row.job_text, k=5)
        msgs = _build_messages(row.job_text, feats, nfr, examples)
        async with sem:
            try:
                resp = await client.chat_completion(
                    messages=msgs, model=LLAMA_MODEL,
                    max_tokens=512, temperature=0.2)
                parsed = _parse_json(resp.choices[0].message.content) or {}
            except Exception as e:
                print(f"  [llama] {type(e).__name__}: {e}")
                parsed = {}
        return {"job_id": int(row.job_id), "fraudulent": int(row.fraudulent), **parsed}

    out = []
    for fut in asyncio.as_completed([task(r) for r in df.itertuples()]):
        out.append(await fut)
        if len(out) % 50 == 0:
            print(f"  llama: {len(out)} / {len(df)}")
    _write("llama_fewshot", out)


def baseline_distilbert(train_csv=TRAIN_CSV, val_csv=VAL_CSV, n_train=2000):
    """B4: DistilBERT on binary fraud label. Maps fraud-prob to JSON.

    Trains on a balanced ~n_train-row subset to keep total runtime
    under ~20 minutes on Mac CPU. The full 11k-row run produces
    similar AUC and would only be useful with a held-out dev split,
    which we don't have separately from val/test.
    """
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from datasets import Dataset
    import torch

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train = pd.read_csv(train_csv)[["job_text", "fraudulent"]].rename(
        columns={"job_text": "text", "fraudulent": "label"})
    fraud = train[train.label == 1].sample(min(n_train // 2, (train.label == 1).sum()),
                                            random_state=42)
    real = train[train.label == 0].sample(n_train - len(fraud), random_state=42)
    train = pd.concat([fraud, real]).sample(frac=1, random_state=42).reset_index(drop=True)

    ds = Dataset.from_pandas(train).map(
        lambda x: tok(x["text"], truncation=True, max_length=512), batched=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(
        output_dir="models/distilbert", num_train_epochs=1,
        per_device_train_batch_size=8, learning_rate=5e-5,
        logging_steps=100, save_strategy="no", report_to="none",
        use_mps_device=False,  # MPS sparse-embedding backward is buggy
        no_cuda=True)
    Trainer(model=mdl, args=args, train_dataset=ds, tokenizer=tok,
            data_collator=DataCollatorWithPadding(tok)).train()

    val = pd.read_csv(val_csv)
    mdl.eval()
    device = next(mdl.parameters()).device
    out = []
    for row in val.itertuples():
        with torch.no_grad():
            ids = tok(row.job_text, return_tensors="pt", truncation=True, max_length=512)
            ids = {k: v.to(device) for k, v in ids.items()}
            p_fraud = torch.softmax(mdl(**ids).logits[0], -1)[1].item()
        out.append({
            "job_id": int(row.job_id), "fraudulent": int(row.fraudulent),
            "trust_score": int(round(100 * (1 - p_fraud))),
            "red_flags": [], "risk_breakdown": {},
            "action": "avoid" if p_fraud > 0.5 else "safe",
            "reasoning": f"DistilBERT fraud probability {p_fraud:.2f}",
        })
    _write("distilbert", out)


def _write(name, rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"baseline_{name}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} predictions to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["b1", "b2", "b3", "b4", "all"], default="b4")
    args = p.parse_args()
    if args.which in ("b1", "all"):
        baseline_phi3("phi3_zeroshot", fewshot=False)
    if args.which in ("b2", "all"):
        baseline_phi3("phi3_fewshot", fewshot=True)
    if args.which in ("b3", "all"):
        asyncio.run(baseline_llama_fewshot())
    if args.which in ("b4", "all"):
        baseline_distilbert()


if __name__ == "__main__":
    main()
