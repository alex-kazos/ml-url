import json
import os
import secrets
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from Services.inference_service import predict_url
from Utilities.config import CHAMPION_FILE_STEM, MODELS_PATH


API_MODEL_NAME = os.getenv("API_MODEL_NAME", CHAMPION_FILE_STEM)
# When set, /metrics requires "Authorization: Bearer <METRICS_TOKEN>".
METRICS_TOKEN = os.getenv("METRICS_TOKEN", "")
# Longer inputs get a 422. Feature extraction and the lookalike check are
# linear or worse in the URL length, and the free tier runs a single worker.
MAX_URL_LENGTH = 2048

DEFAULT_ALLOWED_ORIGINS = (
    "https://alexkazos.com",
    "https://www.alexkazos.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)


def parse_allowed_origins(raw: Optional[str]) -> list[str]:
    """Comma-separated origins from API_ALLOWED_ORIGINS; unset or empty means the defaults."""
    origins = [origin.strip() for origin in (raw or "").split(",") if origin.strip()]
    return origins or list(DEFAULT_ALLOWED_ORIGINS)


ALLOWED_ORIGINS = parse_allowed_origins(os.getenv("API_ALLOWED_ORIGINS"))

app = FastAPI(title="Phishing URL Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTIONS = Counter(
    "mlurl_predictions", "Predictions served, by final verdict.", ["verdict"]
)
LOOKALIKE_OVERRIDES = Counter(
    "mlurl_lookalike_overrides", "Legitimate model verdicts flipped by the lookalike-domain check."
)
PLATFORM_OVERRIDES = Counter(
    "mlurl_platform_overrides",
    "Phishing model verdicts flipped to legitimate by the content-platform rule.",
)
RISK_SIGNALS = Counter(
    "mlurl_risk_signals", "Lookalike risk signals raised, by type.", ["signal_type"]
)
PREDICTION_ERRORS = Counter(
    "mlurl_prediction_errors", "Requests that failed during prediction."
)
PREDICTION_LATENCY = Histogram(
    "mlurl_prediction_latency_seconds",
    "End-to-end prediction latency (features, model and lookalike check).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PHISHING_PROBABILITY = Histogram(
    "mlurl_phishing_probability",
    "Phishing probability returned to users (model-output drift signal).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
INPUT_URL_LENGTH = Histogram(
    "mlurl_input_url_length_chars",
    "Length of submitted URLs (input drift signal).",
    buckets=(20, 40, 60, 80, 120, 160, 240, 400),
)
INPUT_URLS = Counter(
    "mlurl_input_urls",
    "Submitted URLs, by whether they include a scheme (http:// or https://).",
    ["has_scheme"],
)
MODEL_INFO = Gauge(
    "mlurl_model_info",
    "Model currently served (value is always 1).",
    ["model", "version", "algorithm"],
)


def _model_manifest() -> dict:
    """Read Models/<API_MODEL_NAME>.json written by the promotion pipeline, if any."""
    path = MODELS_PATH / f"{API_MODEL_NAME}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


MANIFEST = _model_manifest()
MODEL_VERSION = str(MANIFEST.get("version", "unregistered"))
MODEL_INFO.labels(
    model=API_MODEL_NAME,
    version=MODEL_VERSION,
    algorithm=MANIFEST.get("algorithm", API_MODEL_NAME),
).set(1)


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)


@app.get("/health")
def health():
    return {"status": "ok", "model": API_MODEL_NAME, "version": MODEL_VERSION}


@app.post("/predict")
def predict(payload: PredictRequest):
    INPUT_URL_LENGTH.observe(len(payload.url))
    INPUT_URLS.labels(has_scheme=str("://" in payload.url).lower()).inc()

    start = time.perf_counter()
    try:
        result, details = predict_url(payload.url, model_name=API_MODEL_NAME)
    except Exception:
        PREDICTION_ERRORS.inc()
        raise
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)

    PREDICTIONS.labels(verdict="phishing" if result["label"] == 1 else "legitimate").inc()
    PHISHING_PROBABILITY.observe(result["probability"])
    if details["overridden"]:
        LOOKALIKE_OVERRIDES.inc()
    if details.get("platform_override"):
        PLATFORM_OVERRIDES.inc()
    for signal in result.get("risk_signals", []):
        RISK_SIGNALS.labels(signal_type=signal.split(":", 1)[0]).inc()

    return result


@app.get("/metrics", include_in_schema=False)
def metrics(authorization: Optional[str] = Header(default=None)):
    if METRICS_TOKEN and not secrets.compare_digest(
        (authorization or "").encode(), f"Bearer {METRICS_TOKEN}".encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
