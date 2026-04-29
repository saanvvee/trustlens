# TrustLens — comparison (Kaggle 20-row val subset)

| Model | F1 | Precision | Recall | MAE | Pearson r | ROUGE-L | JSON valid | N |
|-------|-----|-----------|--------|-----|-----------|---------|------------|---|
| baseline_phi3_fewshot | 0.710 | 0.550 | 1.000 | 36.0 | 0.00 | 0.00 | 100% | 20 |
| baseline_phi3_zeroshot | 0.167 | 1.000 | 0.091 | 40.0 | 0.21 | 0.00 | 100% | 20 |
| baseline_distilbert | 0.154 | 0.500 | 0.091 | 44.0 | -0.03 | 0.00 | 100% | 20 |
| finetuned | 0.000 | 0.000 | 0.000 | 44.0 | 0.00 | 0.00 | 100% | 20 |