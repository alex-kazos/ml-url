# Phishing URL Detection

A machine learning pipeline that classifies URLs as phishing or legitimate. The
project includes data extraction, preprocessing, feature engineering, model
training, evaluation plots, command-line inference for a single URL, a FastAPI
service (deployed on Render), MLflow tracking with a model registry, and
Prometheus/Grafana monitoring.

## Project Structure

```text
ml-url/
|-- Classes/                 # Lightweight data containers
|-- Data/                    # Raw and processed datasets, gitignored
|-- Models/                  # Served model bundles, golden set, local artifacts
|-- Notebook/                # Exploration notebooks
|-- Pipelines/               # Training, promotion, retraining and inference entrypoints
|-- Services/                # Pipeline service layer, API, MLflow registry helpers
|-- Tests/                   # Contract tests for labels, features, artifacts and monitoring
|-- Utilities/               # Config and reusable helper functions
|-- monitoring/              # Prometheus config + alert rules, Grafana dashboard
|-- docker-compose.yml       # Local API + Prometheus + Grafana stack
|-- Dockerfile               # Lean API image (same deps as Render)
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
5. Trains Logistic Regression (with scaled inputs), Random Forest, and XGBoost
   baselines.
6. Saves each model as a bundled artifact with:
   - the fitted estimator,
   - the exact feature order,
   - the URL character probability model used for inference,
   - the lookalike-domain reference list,
   - the label mapping.
7. Logs every candidate to MLflow and registers it as a new version of the
   `ml-url` model; the best eligible one gets the `challenger` alias.

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
`Models/`. These files can be large and should normally stay out of Git.

## MLOps: Tracking, Registry and Promotion

MLflow uses a local SQLite database (`mlflow.db`) by default. Recent MLflow
releases no longer accept the old `./mlruns` file store for new runs.

```powershell
# 1. Train: logs every candidate, registers versions, sets the "challenger" alias
python Pipelines\training_pipeline.py

# 2. Compare challenger vs champion on the same held-out URLs, then export
python Pipelines\promotion_pipeline.py

# Or both in one go (e.g. on a weekly schedule)
python Pipelines\retraining_pipeline.py

# Browse runs, metrics and the registry at http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each run logs the usual metrics plus what it costs to serve the model:
`bundle_size_mb`, `inference_ms_p50` and `golden_set_accuracy`, and how many
rows each data source contributed (`data_rows_from_uci`, `data_rows_from_kaggle`).

A challenger only replaces the champion when, on the same evaluation set:

- F1 improves by at least `PROMOTION_MIN_IMPROVEMENT` (default 0.002),
- recall does not drop by more than `PROMOTION_RECALL_TOLERANCE` (default 0.005),
- it scores at least `GOLDEN_SET_MIN_ACCURACY` on `Models/golden_set.csv`,
- its bundle is under `MAX_BUNDLE_MB` (default 50, for Render's free tier).

Promotion exports the champion to `Models/champion.pkl` and
`Models/champion.json`. The API does not talk to MLflow at runtime: commit the
exported files and set `API_MODEL_NAME=champion` on Render to serve them.

Any registered version can also be loaded straight from the registry:

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/ml-url@champion")
model.predict(pd.DataFrame({"url": ["paypa1.com"]}))
```

## Monitoring

The API exposes Prometheus metrics at `/metrics` (set `METRICS_TOKEN` to require
`Authorization: Bearer <token>`):

| Metric | What it tells you |
|:--|:--|
| `mlurl_predictions_total{verdict}` | Traffic and the share of phishing verdicts |
| `mlurl_lookalike_overrides_total` | How often the lookalike rule overrides the model |
| `mlurl_risk_signals_total{signal_type}` | Lookalike / typosquat signals raised |
| `mlurl_prediction_latency_seconds` | End-to-end latency histogram |
| `mlurl_phishing_probability` | Distribution of scores (output drift) |
| `mlurl_input_url_length_chars` | Length of submitted URLs (input drift) |
| `mlurl_input_urls_total{has_scheme}` | URLs typed with or without `http(s)://` |
| `mlurl_model_info{model,version,algorithm}` | Which model version is serving |

Run the API, Prometheus and Grafana locally (needs Docker Desktop):

```powershell
docker compose up --build
python monitoring\simulate_traffic.py --minutes 10            # normal traffic
python monitoring\simulate_traffic.py --minutes 10 --attack   # phishing wave
```

- Grafana: http://localhost:3000 (dashboard "ml-url overview")
- Prometheus: http://localhost:9090 (Alerts tab)

Alert rules live in `monitoring/prometheus/alerts.yml` and have unit tests:

```powershell
cd monitoring\prometheus
promtool test rules alerts_test.yml
```

## Run Inference

```powershell
python Pipelines\inference_pipeline.py
```

By default inference loads `Models/XGBoost.pkl`. New model files are
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
- Promotion gates, per-source row counts and the `/metrics` endpoint behave.
- `Tests/test_model_smoke.py` runs the golden set against the model file the
  API serves, so a mislabelled or stale model fails before it reaches users.
- Rows from both UCI and Kaggle survive the merge with correctly mapped labels
  (UCI uses a lowercase `label` column where 1 = legitimate).
- URLs are normalised (scheme, leading `www.` and trailing `/` removed) before
  any feature is computed, in training and inference alike, so `google.com`
  and `https://www.google.com` get identical features. UCI writes URLs with a
  scheme and Kaggle mostly without, so otherwise the model learns the dataset's
  formatting instead of phishing patterns.

## Data Sources

- [UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- [Kaggle Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
- [Kaggle Top 1M](https://www.kaggle.com/datasets/cheedcheed/top1m)

## Recommended Next Steps

- Add hyperparameter tuning now that label, feature and promotion contracts
  are in place.
- Add host-based signals (domain age, certificate details) for URLs where the
  string alone is not enough.
