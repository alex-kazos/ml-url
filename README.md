# Phishing URL Detection

A machine learning pipeline that classifies URLs as phishing or legitimate. The
project includes data extraction, preprocessing, feature engineering, model
training, evaluation plots, and command-line inference for a single URL.

## Project Structure

```text
ml-url/
|-- Classes/                 # Lightweight data containers
|-- Data/                    # Raw and processed datasets, gitignored
|-- Models/                  # Local model artifacts and plots, gitignored where generated
|-- Notebook/                # Exploration notebooks
|-- Pipelines/               # Training and inference entrypoints
|-- Services/                # Pipeline service layer
|-- Tests/                   # Contract tests for labels, features, and inference artifacts
|-- Utilities/               # Config and reusable helper functions
|-- pyproject.toml           # Tooling/test configuration
|-- requirements.txt
`-- README.md
```

## Current Pipeline

```text
extract_data -> preprocess_data -> merge_data -> feature_engineering -> model_training
```

The training pipeline:

1. Downloads the UCI and Kaggle datasets.
2. Builds URL-derived features.
3. Merges, deduplicates, and balances the datasets.
4. Adds additional URL, domain, path, query, entropy, and keyword features.
5. Trains Logistic Regression, Random Forest, and XGBoost baselines.
6. Saves each model as a bundled artifact with:
   - the fitted estimator,
   - the exact feature order,
   - the URL character probability model used for inference,
   - the label mapping.

The label contract is:

```text
0 = legitimate / safe
1 = phishing / suspicious
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to override data paths or dataset
names.

## Run Training

```powershell
python Pipelines\training_pipeline.py
```

Training writes generated datasets under `Data/` and model artifacts under
`Models/`. These files can be large and should normally stay out of Git. Use
MLflow, DVC, Git LFS, or release assets for sharing trained artifacts.

## Run Inference

```powershell
python Pipelines\inference_pipeline.py
```

By default inference loads `Models/Random_Forest.pkl`. New model files are
metadata bundles rather than raw estimator pickles, so inference can reproduce
the training feature schema.

## Tests

```powershell
pytest
```

The first tests cover the project contracts that are easiest to break:

- `1` means phishing and `0` means legitimate.
- Inference reuses the fitted URL character probability model.
- Model artifacts can carry feature metadata.
- Non-feature columns are dropped before prediction.

## Data Sources

- [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
- [Kaggle Top 1M](https://www.kaggle.com/datasets/cheedcheed/top1m)

## Recommended Next Steps

- Add MLflow logging for params, metrics, plots, and model artifacts.
- Promote the best model through a registry or versioned artifact store.
- Add hyperparameter tuning after the label/feature contracts are stable.
- Build a small Streamlit or FastAPI UI around `inference_service`.
