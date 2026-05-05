# URL Phishing Detection: Model Training Report

This report summarizes one baseline training run for the phishing URL detection
pipeline. Generated plots and model binaries are intentionally ignored by Git;
rerun training to regenerate them locally under `Models/`.

## Dataset Overview

- Total shape: `(228,596, 50)`
- Features used: `44`
- Training set: `182,876` samples
- Test set: `45,720` samples
- Dropped columns: `URL`, `Domain`, `TLD`, `Label`, `TLDLegitimateProb`

## Label Contract

The project uses one label convention throughout training, metrics, and
inference:

```text
0 = legitimate / safe
1 = phishing / suspicious
```

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|:--|--:|--:|--:|--:|--:|--:|
| Random Forest | 0.9026 | 0.8977 | 0.9088 | 0.9032 | 0.9668 | 6.87s |
| XGBoost | 0.8894 | 0.8742 | 0.9097 | 0.8916 | 0.9605 | 0.93s |
| Logistic Regression | 0.7861 | 0.7580 | 0.8406 | 0.7972 | 0.8793 | 10.34s |

## Notes

Random Forest performed best overall in this run, with the strongest F1 score
and ROC-AUC. XGBoost remained close while training faster and achieving slightly
higher recall. Logistic Regression is useful as an interpretable baseline, but
it is weaker for the non-linear URL patterns captured by the tree models.

The generated artifacts from training are:

- `Models/Logistic_Regression.pkl`
- `Models/Random_Forest.pkl`
- `Models/XGBoost.pkl`
- `Models/*_confusion_matrix.png`
- `Models/*_roc_curve.png`
- `Models/*_feature_importance.png`

For reproducible experiment tracking, prefer MLflow, DVC, Git LFS, release
assets, or another artifact store instead of committing these files directly.

[Go to README](../README.md)
