"""MLflow tracking and Model Registry helpers for ml-url.

The workflow is champion / challenger:

1. Training logs every candidate as a pyfunc model that wraps the *whole*
   bundle, and registers it as a new version of one registered model
   (``MLFLOW_MODEL_NAME``, default ``ml-url``).
2. The best eligible candidate gets the ``challenger`` alias.
3. Promotion re-scores ``challenger`` and ``champion`` on the same fresh
   evaluation set, through the same code path the API uses, and only moves the
   ``champion`` alias when the challenger clears the acceptance criteria.
4. The champion is exported to ``Models/champion.pkl`` (+ ``champion.json``)
   for the Render API, which never talks to MLflow at runtime.
"""
import json
import os
import pickle
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Services.inference_service import predict_with_bundle
from Utilities.config import (
    CHAMPION_FILE_STEM,
    EVAL_SAMPLE_SIZE,
    GOLDEN_SET_MIN_ACCURACY,
    GOLDEN_SET_PATH,
    MAX_BUNDLE_MB,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_REGISTRY_URI,
    MLFLOW_TRACKING_URI,
    MODELS_PATH,
    PROJECT_ROOT,
    PROMOTION_METRIC,
    PROMOTION_MIN_IMPROVEMENT,
    PROMOTION_RECALL_TOLERANCE,
)

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
EVAL_SET_ARTIFACT = "eval/eval_set.csv"
PYFUNC_MODEL_PATH = PROJECT_ROOT / "Services" / "pyfunc_model.py"
CODE_PATHS = [str(PROJECT_ROOT / name) for name in ("Services", "Utilities", "Classes")]
SERVING_REQUIREMENTS = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure_mlflow() -> MlflowClient:
    """Point MLflow at the configured tracking/registry backend and experiment."""
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return MlflowClient()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 with 1 = phishing as the positive class."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> pd.DataFrame:
    """Hand-picked URLs with known labels (1 = phishing, 0 = legitimate)."""
    golden = pd.read_csv(path, comment="#")
    return golden[["url", "label"]].astype({"url": str, "label": int})


def golden_set_check(
    predict_labels: Callable[[List[str]], List[int]],
    golden: Optional[pd.DataFrame] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Return (accuracy, failures) of a model on the golden set.

    ``predict_labels`` takes a list of URLs and returns final labels, so the
    check runs through exactly the code path that serves users.
    """
    golden = load_golden_set() if golden is None else golden
    urls = golden["url"].tolist()
    expected = golden["label"].tolist()
    predicted = [int(label) for label in predict_labels(urls)]

    failures = [
        {"url": url, "expected": exp, "predicted": pred}
        for url, exp, pred in zip(urls, expected, predicted)
        if exp != pred
    ]
    accuracy = 1 - len(failures) / len(urls) if urls else 0.0
    return accuracy, failures


def bundle_predictor(model: Any, metadata: Dict[str, Any]) -> Callable[[List[str]], List[int]]:
    """Wrap an in-memory bundle as a ``urls -> labels`` function."""
    def _predict(urls: List[str]) -> List[int]:
        return [predict_with_bundle(url, model, metadata)[0]["label"] for url in urls]
    return _predict


def measure_latency_ms(
    model: Any,
    metadata: Dict[str, Any],
    urls: Sequence[str],
) -> float:
    """Median end-to-end latency (features + model + lookalike check) per URL."""
    timings = []
    for url in urls:
        start = time.perf_counter()
        predict_with_bundle(url, model, metadata)
        timings.append((time.perf_counter() - start) * 1000)
    return float(statistics.median(timings)) if timings else 0.0


def build_eval_set(
    urls: pd.Series,
    labels: pd.Series,
    sample_size: int = EVAL_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified sample of held-out URLs used to compare champion vs challenger."""
    eval_df = pd.DataFrame({"url": urls.astype(str).values, "label": labels.astype(int).values})
    if len(eval_df) <= sample_size:
        return eval_df.reset_index(drop=True)

    fraction = sample_size / len(eval_df)
    sampled = eval_df.groupby("label", group_keys=False).sample(
        frac=fraction, random_state=random_state
    )
    return sampled.reset_index(drop=True)


def to_output_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a prediction result into the pyfunc output schema."""
    return {
        "url": result["url"],
        "label": int(result["label"]),
        "probability": float(result["probability"]),
        "verdict": result["verdict"],
        "risk_signals": ",".join(result.get("risk_signals", [])),
    }


def load_version_bundle(version: Any) -> Tuple[Any, Dict[str, Any]]:
    """Download a registered version's bundle and return (model, metadata)."""
    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{MLFLOW_MODEL_NAME}/{version.version}", dst_path=tmp
            )
        )
        bundles = sorted(local_dir.rglob("artifacts/*.pkl"))
        if not bundles:
            raise FileNotFoundError(f"No bundle in version {version.version} at {local_dir}")
        with open(bundles[0], "rb") as f:
            # Only unpickle bundles from your own registry.
            artifact = pickle.load(f)
    model = artifact["model"]
    metadata = {k: v for k, v in artifact.items() if k != "model"}
    return model, metadata


# ---------------------------------------------------------------------------
# Logging and registering candidates
# ---------------------------------------------------------------------------

def log_bundle_model(
    bundle_path: Path,
    model: Any,
    metadata: Dict[str, Any],
    algorithm: str,
) -> Any:
    """Log the full bundle as a pyfunc model and register a new version.

    Must be called inside an active MLflow run.
    """
    sample_in = pd.DataFrame({"url": ["https://www.google.com", "paypa1.com"]})
    sample_out = pd.DataFrame(
        [to_output_row(predict_with_bundle(url, model, metadata)[0]) for url in sample_in["url"]]
    )

    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=str(PYFUNC_MODEL_PATH),
        artifacts={"bundle": str(bundle_path)},
        code_paths=CODE_PATHS,
        signature=infer_signature(sample_in, sample_out),
        pip_requirements=SERVING_REQUIREMENTS,
        registered_model_name=MLFLOW_MODEL_NAME,
    )

    client = MlflowClient()
    version = info.registered_model_version
    client.set_model_version_tag(MLFLOW_MODEL_NAME, version, "algorithm", algorithm)
    client.set_model_version_tag(MLFLOW_MODEL_NAME, version, "bundle_file", bundle_path.name)
    return info


def set_alias(alias: str, version: str) -> None:
    MlflowClient().set_registered_model_alias(MLFLOW_MODEL_NAME, alias, str(version))


def get_version_by_alias(alias: str) -> Optional[Any]:
    try:
        return MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, alias)
    except mlflow.exceptions.MlflowException:
        return None


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionCriteria:
    metric: str = PROMOTION_METRIC
    min_improvement: float = PROMOTION_MIN_IMPROVEMENT
    recall_tolerance: float = PROMOTION_RECALL_TOLERANCE
    max_bundle_mb: float = MAX_BUNDLE_MB
    golden_min_accuracy: float = GOLDEN_SET_MIN_ACCURACY


def promotion_decision(
    champion: Optional[Dict[str, float]],
    challenger: Dict[str, float],
    criteria: PromotionCriteria = PromotionCriteria(),
) -> Tuple[bool, List[str]]:
    """Decide whether the challenger replaces the champion.

    Both dicts hold metrics measured on the *same* evaluation set, plus
    ``golden_set_accuracy`` and ``bundle_size_mb``. Returns (promote, reasons).
    """
    reasons: List[str] = []

    if challenger["golden_set_accuracy"] < criteria.golden_min_accuracy:
        reasons.append(
            f"golden set accuracy {challenger['golden_set_accuracy']:.2f} "
            f"< {criteria.golden_min_accuracy:.2f}"
        )
    if challenger["bundle_size_mb"] > criteria.max_bundle_mb:
        reasons.append(
            f"bundle {challenger['bundle_size_mb']:.1f} MB > {criteria.max_bundle_mb:.0f} MB limit"
        )

    if champion is None:
        if reasons:
            return False, reasons
        return True, ["no current champion and the challenger passes every gate"]

    gain = challenger[criteria.metric] - champion[criteria.metric]
    if gain < criteria.min_improvement:
        reasons.append(
            f"{criteria.metric} gain {gain:+.4f} < required {criteria.min_improvement:+.4f}"
        )
    recall_drop = champion["Recall"] - challenger["Recall"]
    if recall_drop > criteria.recall_tolerance:
        reasons.append(
            f"recall drops by {recall_drop:.4f} (> {criteria.recall_tolerance:.4f} allowed)"
        )

    if reasons:
        return False, reasons
    return True, [f"{criteria.metric} improves by {gain:+.4f} and all gates pass"]


def _run_metric(run_id: str, key: str, default: float = 0.0) -> float:
    return float(MlflowClient().get_run(run_id).data.metrics.get(key, default))


def score_version(version: Any, eval_df: pd.DataFrame) -> Dict[str, float]:
    """Re-score a registered version on the eval set and the golden set.

    Uses the same ``predict_with_bundle`` path as the API (features, model and
    lookalike override), so the numbers describe what users would get.
    """
    model, metadata = load_version_bundle(version)
    predict = bundle_predictor(model, metadata)
    metrics = classification_metrics(eval_df["label"].tolist(), predict(eval_df["url"].tolist()))
    golden_accuracy, _ = golden_set_check(predict)
    metrics["golden_set_accuracy"] = golden_accuracy
    metrics["bundle_size_mb"] = _run_metric(version.run_id, "bundle_size_mb")
    return metrics


def promote_challenger(criteria: PromotionCriteria = PromotionCriteria()) -> Dict[str, Any]:
    """Compare challenger vs champion on the same fresh data and maybe promote."""
    configure_mlflow()
    challenger = get_version_by_alias(CHALLENGER_ALIAS)
    if challenger is None:
        raise RuntimeError("No model version has the 'challenger' alias. Train first.")
    champion = get_version_by_alias(CHAMPION_ALIAS)
    if champion is not None and champion.version == challenger.version:
        print(f"v{challenger.version} is already the champion; nothing to do.")
        return {"promoted": False, "reasons": ["challenger is already champion"],
                "challenger_version": challenger.version, "champion_version": champion.version}

    eval_path = mlflow.artifacts.download_artifacts(
        run_id=challenger.run_id, artifact_path=EVAL_SET_ARTIFACT
    )
    eval_df = pd.read_csv(eval_path)
    print(f"Evaluation set: {len(eval_df):,} held-out URLs from run {challenger.run_id}")

    print(f"Scoring challenger v{challenger.version} ...")
    challenger_metrics = score_version(challenger, eval_df)
    champion_metrics = None
    if champion is not None:
        print(f"Scoring champion   v{champion.version} ...")
        champion_metrics = score_version(champion, eval_df)

    promote, reasons = promotion_decision(champion_metrics, challenger_metrics, criteria)

    with mlflow.start_run(run_name=f"promotion: v{challenger.version}"):
        mlflow.set_tag("stage", "promotion")
        mlflow.log_params(
            {
                "challenger_version": challenger.version,
                "champion_version": champion.version if champion else "none",
                "decision": "promote" if promote else "keep",
                **{f"criteria_{k}": v for k, v in asdict(criteria).items()},
            }
        )
        mlflow.log_metrics({f"challenger_{k}": v for k, v in challenger_metrics.items()})
        if champion_metrics:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_text("\n".join(reasons), "decision.txt")

    if promote:
        set_alias(CHAMPION_ALIAS, challenger.version)
        print(f"Promoted v{challenger.version} to champion: {'; '.join(reasons)}")
    else:
        print(f"Kept current champion: {'; '.join(reasons)}")

    return {
        "promoted": promote,
        "reasons": reasons,
        "challenger_version": challenger.version,
        "champion_version": challenger.version if promote else (champion.version if champion else None),
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
    }


# ---------------------------------------------------------------------------
# Export for the Render API
# ---------------------------------------------------------------------------

def export_champion(
    models_dir: Path = MODELS_PATH,
    file_stem: str = CHAMPION_FILE_STEM,
) -> Path:
    """Copy the champion bundle to ``Models/<file_stem>.pkl`` plus a JSON manifest."""
    configure_mlflow()
    champion = get_version_by_alias(CHAMPION_ALIAS)
    if champion is None:
        raise RuntimeError("No model version has the 'champion' alias yet.")

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{MLFLOW_MODEL_NAME}@{CHAMPION_ALIAS}", dst_path=tmp
            )
        )
        bundles = sorted(local_dir.rglob("artifacts/*.pkl"))
        if not bundles:
            raise FileNotFoundError(f"No bundle found in champion artifacts at {local_dir}")

        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        destination = models_dir / f"{file_stem}.pkl"
        shutil.copy2(bundles[0], destination)

    run_metrics = MlflowClient().get_run(champion.run_id).data.metrics
    manifest = {
        "registered_model": MLFLOW_MODEL_NAME,
        "version": str(champion.version),
        "algorithm": champion.tags.get("algorithm", "unknown"),
        "run_id": champion.run_id,
        "metrics": {k: round(v, 4) for k, v in sorted(run_metrics.items())},
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (models_dir / f"{file_stem}.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported champion v{champion.version} ({manifest['algorithm']}) -> {destination}")
    return destination
