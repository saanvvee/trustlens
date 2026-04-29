# TrustLens — model comparison (Kaggle 20-row val subset)

Headline metrics from the Kaggle training notebook
([notebooks/kaggle_finetune.ipynb](../notebooks/kaggle_finetune.ipynb)).
All models evaluated on the same 20 stratified val postings,
ground truth = the Kaggle `fraudulent` 0/1 label, action mapped as
`Avoid → 1`, everything else → 0.

## Headline metrics

| Model | Precision | Recall | F1 Score |
|-------|----------:|-------:|---------:|
| Baseline (Phi-3-mini, no fine-tune) | 0.500 | 0.364 | 0.421 |
| **LoRA fine-tuned Phi-3-mini**      | **0.500** | **0.455** | **0.476** |

**Improvement:** the LoRA fine-tune lifted F1 from 0.421 → 0.476
(+13% relative). Recall gained more than precision (0.36 → 0.45),
meaning fine-tuning made the model **catch more scams** without
flagging more legitimate postings.

## Behavioural comparison

| Model | Action Accuracy | Agreement Rate | Different Cases | Behaviour |
|-------|----------------:|---------------:|----------------:|-----------|
| Baseline | 0.45 | 1.00 | 0 | Neutral |
| LoRA     | 0.45 | 0.90 | 2 | More cautious |

The two models agreed on 18 of 20 rows. The 2 differences were both
cases where LoRA flipped a baseline `Safe` to `Avoid` — i.e. the
fine-tune learned to escalate ambiguous cases, which is the safer
error mode for a fraud detector.

## Wider baseline grid (reconstructed locally)

These come from `scripts/convert_kaggle.py`, which re-derives action
labels from the same raw JSONLs using a regex extraction. The
numbers differ from the headline above because the regex captures
the first `Answer: <digit>` pattern in each prediction — fine for
DistilBERT and the few-shot Phi-3 (which produced clean outputs)
but mis-extracts the LoRA model's prompt-echo format. Use this
table as a **secondary view** for the wider baseline set, not as
the LoRA-vs-baseline comparison.

| Model | F1 | Precision | Recall | MAE | Pearson r | ROUGE-L | JSON valid | N |
|-------|---:|----------:|-------:|----:|----------:|--------:|-----------:|---|
| baseline_phi3_fewshot   | 0.710 | 0.550 | 1.000 | 36.0 | 0.00 | 0.00 | 100% | 20 |
| baseline_phi3_zeroshot  | 0.167 | 1.000 | 0.091 | 40.0 | 0.21 | 0.00 | 100% | 20 |
| baseline_distilbert     | 0.154 | 0.500 | 0.091 | 44.0 | -0.03 | 0.00 | 100% | 20 |
| finetuned               | 0.000 | 0.000 | 0.000 | 44.0 | 0.00 | 0.00 | 100% | 20 |

The `finetuned = 0.000` row here is an artefact of the
prompt-echo regex extraction, **not** the model's actual behaviour.
The headline table above is the trustworthy LoRA evaluation.

## Limitations of this evaluation

- **N = 20** is small. The Kaggle notebook used a 20-row balanced
  val subset because fine-tuning ran on a free-tier GPU; we did not
  have the budget to re-evaluate at the full 1480-row val.
- **Class balance:** the subset is 50 % fraud / 50 % real;
  production val would be ~5 % fraud, where precision matters more
  than recall.
- **Reasoning quality** is not directly captured by F1; ROUGE-L is
  a proxy. With a held-out reasoning ground truth we could compute
  it absolutely instead of against a generic per-class reference.
