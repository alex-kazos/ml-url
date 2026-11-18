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

1. Downloads the UCI and Kaggle datasets into `Data/raw`. If the UCI site
   cannot be reached, the copy saved there by an earlier run is reused (with a
   warning); the UCI dataset is a fixed snapshot, so nothing is lost.
2. Builds URL-derived features.
3. Merges, deduplicates, and balances the datasets. URLs that appear labelled
   both legitimate and phishing (across or within sources) are dropped.
4. Adds additional URL, domain, path, query, entropy, and keyword features.
   Brand names and "pay" only count outside the registered domain or in a
   domain the brand does not own (`paypal.com` itself is not suspicious,
   `paypal.com.account-verify.xyz` and `paypal-secure.com` are), and "@" only
   counts as an email address or as user info before the host, not as a
   profile handle (`substack.com/@name`).
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

# Reuse the files already in Data/raw instead of downloading them again
python Pipelines\training_pipeline.py --skip-extract
```

Training writes generated datasets under `Data/` and model artifacts under
`Models/`. These files can be large and should normally stay out of Git.
Each run also rewrites `Models/Model_Comparison.md`, a summary of the run
(rows per source, features, metrics per candidate, the challenger) for readers
who do not have access to MLflow.

Training refuses a dataset whose `label_binary` disagrees with its `Label`
column (1 must mean `Label == "bad"`), so a stale `ml_ready_dataset.csv` from an
older label convention fails loudly instead of training a broken model.

## MLOps: Tracking, Registry and Promotion

MLflow uses a local SQLite database (`mlflow.db`) by default. Recent MLflow
releases no longer accept the old `./mlruns` file store for new runs.

```powershell
# 1. Train: logs every candidate, registers versions, sets the "challenger" alias
python Pipelines\training_pipeline.py

# 2. Compare challenger vs champion on the frozen holdout set, then export
python Pipelines\promotion_pipeline.py

# Or both in one go (e.g. on a weekly schedule)
python Pipelines\retraining_pipeline.py

# Browse runs, metrics and the registry at http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each run logs the usual metrics plus what it costs to serve the model:
`bundle_size_mb`, `inference_ms_p50` and `golden_set_accuracy`, and how many
rows each data source contributed (`data_rows_from_uci`, `data_rows_from_kaggle`,
`data_rows_in_both`, `data_urls_with_conflicting_labels`).

Champion and challenger are compared on `Models/holdout_set.csv`: 2,000
labelled URLs (half phishing) frozen from the v9 run's test split. Every
training run removes them from its data before the train/test split, so no
model is ever scored on URLs it trained on, and every promotion uses the same
yardstick. The URL character model is fitted on the training split only.

Every challenger must pass absolute gates:

- it scores at least `GOLDEN_SET_MIN_ACCURACY` (default 0.75) on `Models/golden_set.csv`,
- its precision is at least `PROMOTION_MIN_PRECISION` (default 0.80),
- its bundle is under `MAX_BUNDLE_MB` (default 50, for Render's free tier).

It then replaces the champion when, on the same evaluation set:

- F1 improves by at least `PROMOTION_MIN_IMPROVEMENT` (default 0.002), and
- recall does not drop by more than `PROMOTION_RECALL_TOLERANCE` (default 0.005).

If the champion itself fails the golden-set or precision gate, any challenger
that passes the absolute gates replaces it. Otherwise a model that calls almost
everything phishing (recall near 1, precision near 0.5) could block a better
one through the recall gate forever.

Promotion exports the champion to `Models/champion.pkl` and
`Models/champion.json`. The API does not talk to MLflow at runtime: commit the
exported files to serve them. The API, Docker Compose, the CLI and the smoke
test all default to `champion`; `Models/XGBoost.pkl` and the other candidate
bundles are overwritten by every training run, promoted or not.

`requirements-api.txt` pins pandas, numpy, scikit-learn and xgboost to the
versions of the training environment, because the bundle is a pickle and only
loads reliably with the versions that wrote it. Update the pins (and
`SERVING_REQUIREMENTS` in `Services/model_registry.py`, which a test keeps in
step) whenever a champion is trained with newer libraries.

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

By default inference loads `Models/champion.pkl`, the model the API serves.
New model files are metadata bundles rather than raw estimator pickles, so
inference can reproduce the training feature schema.

On top of the model, a lookalike rule flags domains that imitate a well-known
one. Leetspeak copies (`paypa1.com`) and known sites are checked against the
top 100,000 domains of `top-1m.csv`; typosquats (up to two edits away, such as
`linkedln.com`) only against the top `LOOKALIKE_TARGET_LIMIT` (default 5,000).
Matching an unknown site against all 100,000 always finds some obscure domain
two edits away (`substack.com` -> `tubestack.com`), and nobody typosquats those.

`/predict` accepts URLs of 1 to 2,048 characters; anything else gets a 422.
CORS origins come from `API_ALLOWED_ORIGINS` (comma-separated); when it is
unset, the API allows `alexkazos.com`, `www.alexkazos.com` and
`localhost:8080`.

## Tests

```powershell
pytest
```

CI (`.github/workflows/ci.yml`) runs on every pull request and every push to
`master`: `ruff check .`, the test suite (including the smoke tests on the
committed `Models/champion.pkl`) with the dependencies in
`requirements-ci.txt`, and `promtool` on the alert rules.

The first tests cover the project contracts that are easiest to break:

- `1` means phishing and `0` means legitimate.
- Inference reuses the fitted URL character probability model.
- Model artifacts can carry feature metadata.
- Non-feature columns are dropped before prediction.
- Promotion gates, per-source row counts and the `/metrics` endpoint behave.
- `Tests/test_model_smoke.py` runs the golden set against the model file the
  API serves, so a mislabelled or stale model fails before it reaches users,
  and checks with the bundle's own domain list that the lookalike rule leaves
  legitimate sites alone (`substack.com`, `sfnmjournal.com`) but still catches
  typosquats (`linkedln.com`, `faceb00k.com`).
- Training rejects a dataset whose `label_binary` disagrees with `Label`.
- A champion that fails its own gates cannot block a healthy challenger.
- The serving pins in `requirements-api.txt` match the ones logged with every
  MLflow model.
- Inference computes every feature a served bundle lists, including the
  retired ones older bundles still use, so none silently becomes 0.
- New models never see retired features; the character model is fitted on
  the training split only; URLs labelled both ways are dropped; the holdout
  set stays out of training.
- Brand, pay and "@" features ignore a brand's own domain and profile handles.
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

- The bare homepage of a user subdomain on a platform
  (`alexkazos.substack.com`, `name.medium.com`) still scores as phishing
  (about 0.65); post URLs on the same host score below 0.03. In the training
  data, UCI's legitimate rows never have a path, and hosts with a subdomain
  but no path are 77% phishing, mostly real phishing on free hosting
  (`xxx.web.app`, `xxx.firebaseapp.com`). From the string alone a reputable
  platform's user subdomain looks the same. Fixing it needs information
  beyond the string, such as the registered domain's popularity rank from a
  newer list than `top-1m.csv` (which does not contain `substack.com`), or
  legitimate platform-subdomain URLs in the training data.
- `Bank` and `Crypto` still fire on domains that own the word
  (`bankofamerica.com`, `crypto.com`), as `Pay` did before v21.
- Once no served bundle lists them, delete the retired features
  (`RETIRED_FEATURES` in `Services/model_training.py`) from the feature code.
- Add hyperparameter tuning now that label, feature and promotion contracts
  are in place.
- Add host-based signals (domain age, certificate details) for URLs where the
  string alone is not enough.
