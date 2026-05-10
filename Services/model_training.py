import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import mlflow
import mlflow.sklearn


from Utilities.config import MODELS_PATH, PROCESSED_DATA_PATH, MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_URI
from Utilities.Services.preprocess_data_utils import (
    URLCharacterProbabilityModel,
    apply_url_char_probability_model,
)
from Services.inference_service import _load_reference_domains, _reference_domain_index


warnings.filterwarnings("ignore")

COLUMNS_TO_DROP: List[str] = [
    "URL",
    "Domain",
    "TLD",
    "Label",
    "TLDLegitimateProb",
]

TARGET_COLUMN = "label_binary"
RANDOM_STATE = 42
LABEL_MAPPING = {"legitimate": 0, "phishing": 1}


try:
    from sklearn.model_selection import train_test_split as _sklearn_train_test_split
except ImportError:
    _sklearn_train_test_split = None


def _train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if _sklearn_train_test_split is not None:
        return _sklearn_train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    test_indices = []
    for _, group in y.groupby(y):
        test_count = max(1, round(len(group) * test_size))
        test_indices.extend(
            group.sample(n=test_count, random_state=random_state).index.tolist()
        )

    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]
    X_train = X.drop(index=test_indices)
    y_train = y.drop(index=test_indices)
    return X_train, X_test, y_train, y_test


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the ML-ready CSV dataset."""
    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")
    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Optional[URLCharacterProbabilityModel],
]:
    """Drop non-feature columns and split into train / test."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    url_char_model = (
        URLCharacterProbabilityModel.fit(df["URL"]) if "URL" in df.columns else None
    )
    if url_char_model is not None:
        df = apply_url_char_probability_model(
            df.copy(),
            df["URL"].astype(str),
            url_char_model,
        )

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped columns: {cols_to_drop}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = _train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test  set: {X_test.shape[0]:,} samples")
    print(f"  Features:  {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, url_char_model


def train_model(name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit model and return it."""
    print(f"\n{'='*60}")
    print(f" Training: {name}")
    print(f"{'='*60}")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.2f}s")
    return model


def evaluate(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path = MODELS_PATH,
) -> Dict[str, float]:
    """Evaluate a trained model, print report, save plots."""
    from Utilities.Services.model_training_utils import (
        evaluate_model,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_roc_curve,
        print_classification_report,
    )

    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        phishing_index = classes.index(1) if 1 in classes else 1
        y_proba = model.predict_proba(X_test)[:, phishing_index]

    metrics = evaluate_model(y_test, y_pred, y_proba)
    print(f"\n  Metrics for {name}:")
    for k, v in metrics.items():
        print(f"    {k:<12s}: {v:.4f}")

    print_classification_report(y_test, y_pred)

    safe_name = name.replace(" ", "_")
    plot_confusion_matrix(
        y_test,
        y_pred,
        name,
        models_dir / f"{safe_name}_confusion_matrix.png",
    )
    if y_proba is not None:
        plot_roc_curve(
            y_test,
            y_proba,
            name,
            models_dir / f"{safe_name}_roc_curve.png",
        )
    plot_feature_importance(
        model,
        list(X_test.columns),
        name,
        models_dir / f"{safe_name}_feature_importance.png",
    )
    return metrics


def save_model(
    name: str,
    model: Any,
    feature_names: List[str],
    url_char_model: Optional[URLCharacterProbabilityModel] = None,
    reference_domains: Optional[List[str]] = None,
    models_dir: Path = MODELS_PATH,
) -> Path:
    """Pickle the trained model with the metadata needed for inference."""
    safe_name = name.replace(" ", "_")
    path = models_dir / f"{safe_name}.pkl"
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "url_char_model": url_char_model,
        "reference_domains": tuple(reference_domains or ()),
        "reference_domain_index": _reference_domain_index(tuple(reference_domains or ())),
        "label_mapping": LABEL_MAPPING,
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Saved model -> {path}")
    return path


def _print_summary(results: Dict[str, Dict[str, float]]) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*70}")
    print(" MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':<25s}"
    metric_names = list(next(iter(results.values())).keys())
    for metric_name in metric_names:
        header += f"{metric_name:>12s}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:<25s}"
        for metric_name in metric_names:
            row += f"{metrics.get(metric_name, 0):>12.4f}"
        print(row)
    print(f"{'='*70}")


def model_training_service(
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Execute the complete training pipeline and return model metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    if df is not None:
        print(f"Using provided DataFrame (shape: {df.shape})")
    else:
        if data_path is None:
            data_path = PROCESSED_DATA_PATH / "ml_ready_dataset.csv"
        df = load_data(data_path)

    if models_dir is None:
        models_dir = MODELS_PATH

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, url_char_model = prepare_data(
        df,
        test_size,
        random_state,
    )
    feature_names = list(X_train.columns)
    reference_domains = list(_load_reference_domains())

    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }

    results: Dict[str, Dict[str, float]] = {}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow registry URI: {mlflow.get_registry_uri()}")

    mlflow.set_experiment("Phishing URL Detection")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)

            params = model.get_params()
            mlflow.log_params({
                k: v for k, v in params.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            })

            trained = train_model(name, model, X_train, y_train)
            metrics = evaluate(name, trained, X_test, y_test, models_dir)

            mlflow.log_metrics(metrics)

            model_path = save_model(
                name,
                trained,
                feature_names,
                url_char_model,
                reference_domains,
                models_dir,
            )
            mlflow.log_artifact(str(model_path), artifact_path="model_bundle")

            safe_name = name.replace(" ", "_")
            for artifact_name in [
                f"{safe_name}_confusion_matrix.png",
                f"{safe_name}_roc_curve.png",
                f"{safe_name}_feature_importance.png",
            ]:
                artifact_path = models_dir / artifact_name
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path), artifact_path="plots")

            mlflow.sklearn.log_model(
                trained,
                artifact_path="sklearn_model",
                registered_model_name=name.replace(" ", "_"),
            )

            results[name] = metrics


if __name__ == "__main__":
    model_training_service()
