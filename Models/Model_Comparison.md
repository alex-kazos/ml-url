# URL Phishing Detection: Model Comparison

Written by `Services/model_training.py` at the end of every training run, so it
describes the latest run. Do not edit it by hand. The model the API serves is
the promoted champion in `Models/champion.json`, which can come from an
earlier run.

Generated: 2026-09-30T16:42:05+00:00

## Dataset

- Rows: 411,080
- From UCI: 149,677, from Kaggle: 261,982, in both: 579
- Features: 43
- Training set: 328,864
- Test set: 82,216
- Not used as features: `URL`, `Domain`, `TLD`, `Label`, `TLDLegitimateProb`

## Candidates

Accuracy to ROC-AUC are measured on the test split with the model alone. The
golden set (`Models/golden_set.csv`) goes through the full prediction path,
lookalike rule included, like the API.

| Model | Version | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Golden set | Bundle (MB) | p50 (ms) | Training (s) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | v7 | 0.7860 | 0.8313 | 0.7177 | 0.7703 | 0.8659 | 0.8750 | 8.2 | 13.6 | 1.5 |
| Random Forest | v8 | 0.8603 | 0.8675 | 0.8505 | 0.8589 | 0.9370 | 0.8750 | 695.3 | 53.9 | 15.0 |
| XGBoost | v9 | 0.8684 | 0.9083 | 0.8194 | 0.8616 | 0.9438 | 0.9375 | 8.6 | 71.7 | 4.0 |

Challenger: **XGBoost** (v9), the best F1-Score among the candidates that pass the golden-set, precision and size gates. `Pipelines/promotion_pipeline.py` compares it with the champion on the same evaluation set before anything is served.

## Label Contract

```text
0 = legitimate / safe
1 = phishing / suspicious
```

[Go to README](../README.md)
