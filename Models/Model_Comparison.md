# URL Phishing Detection: Model Comparison

Written by `Services/model_training.py` at the end of every training run, so it
describes the latest run. Do not edit it by hand. The model the API serves is
the promoted champion in `Models/champion.json`, which can come from an
earlier run.

Generated: 2026-11-18T08:09:31+00:00

## Dataset

- Rows: 409,650
- From UCI: 149,637, from Kaggle: 261,937, in both: 549
- Holdout URLs kept out of training: 1,380 (`Models/holdout_set.csv`, used for promotion)
- Features: 40
- Training set: 327,720
- Test set: 81,930
- Not used as features: `URL`, `Domain`, `TLD`, `Label`, `TLDLegitimateProb`
- Retired (still computed for older bundles): `HasTitle`, `NoOfSubDomain`, `HasObfuscation`, `URLSimilarityIndex`, `Pay`, `HasBrandName`, `NumAtSymbol`, `NoOfObfuscatedChar`, `ObfuscationRatio`

## Candidates

Accuracy to ROC-AUC are measured on the test split with the model alone. The
golden set (`Models/golden_set.csv`) goes through the full prediction path,
lookalike rule included, like the API.

| Model | Version | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Golden set | Bundle (MB) | p50 (ms) | Training (s) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | v19 | 0.7850 | 0.8315 | 0.7142 | 0.7684 | 0.8666 | 0.9048 | 8.2 | 14.1 | 1.3 |
| Random Forest | v20 | 0.8546 | 0.8571 | 0.8505 | 0.8538 | 0.9308 | 0.8095 | 707.6 | 65.5 | 16.5 |
| XGBoost | v21 | 0.8661 | 0.9047 | 0.8181 | 0.8592 | 0.9412 | 0.9524 | 8.6 | 31.6 | 1.8 |

Challenger: **XGBoost** (v21), the best F1-Score among the candidates that pass the golden-set, precision and size gates. `Pipelines/promotion_pipeline.py` compares it with the champion on the same evaluation set before anything is served.

## Label Contract

```text
0 = legitimate / safe
1 = phishing / suspicious
```

[Go to README](../README.md)
