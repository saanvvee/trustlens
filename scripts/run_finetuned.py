"""Run the fine-tuned Phi-3 + LoRA adapter on the val set.

Produces ``data/labeled/finetuned.jsonl`` — the file ``src.evaluate``
treats as the fine-tuned-model column in the comparison table.

Run on Colab (needs CUDA + bitsandbytes; will not run on Mac).

Usage (inside Colab, after fine-tuning):
    python -m scripts.run_finetuned \\
        --adapter /content/drive/MyDrive/trustlens/phi3-trustlens-lora
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.features import extract_all
from src.label_generator import SYSTEM_PROMPT, _parse_json, build_user_prompt
from src.vector_store import VectorStore

VAL_CSV = "data/processed/val.csv"
OUT = Path("data/labeled/finetuned.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True,
                   help="path to LoRA adapter dir on Drive")
    p.add_argument("--val", default=VAL_CSV)
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    mdl = PeftModel.from_pretrained(base, args.adapter)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.eval()

    df = pd.read_csv(args.val)
    vs = VectorStore()
    out = []
    for i, row in enumerate(df.itertuples()):
        feats = extract_all(row.job_text)
        nfr = vs.neighbor_fraud_rate(row.job_text, k=5)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row.job_text, feats, nfr)},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3000).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**ids, max_new_tokens=512, do_sample=False)
        text = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = _parse_json(text) or {}
        out.append({"job_id": int(row.job_id), "fraudulent": int(row.fraudulent), **parsed})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(df)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} predictions to {OUT}")


if __name__ == "__main__":
    main()
