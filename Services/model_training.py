import pickle
import time
from pathlib import Path
from typing import (
Any,
Dict,
List,
Tuple
)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# warnings == success
import warnings
warnings.filterwarnings("ignore")

# Import configuration
from Utilities.config import MODELS_PATH, PROCESSED_DATA_PATH

# Import utility functions
from Utilities.Services.model_training_utils import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    print_classification_report,
)

# Columns that are non-numeric or introduce data leakage
COLUMNS_TO_DROP: List[str] = [
    "URL",
    "Domain",
    "TLD",
    "Label",
    "TLDLegitimateProb",
]

TARGET_COLUMN: str = "label_binary"
RANDOM_STATE: int = 42


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the ML-ready CSV dataset."""
    print(f"Loading data from {data_path} …")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")
    return df


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Drop non-feature columns and split into train / test."""
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped columns: {cols_to_drop}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test  set: {X_test.shape[0]:,} samples")
    print(f"  Features:  {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def train_model(name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit *model* and return it."""
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
    y_pred = model.predict(X_test)

    # Probability estimates for ROC-AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print(f"\n  Metrics for {name}:")
    for k, v in metrics.items():
        print(f"    {k:<12s}: {v:.4f}")

    # Classification report
    print_classification_report(y_test, y_pred)

    # Plots
    safe_name = name.replace(" ", "_")
    plot_confusion_matrix(
        y_test, y_pred, name,
        models_dir / f"{safe_name}_confusion_matrix.png",
    )
    if y_proba is not None:
        plot_roc_curve(
            y_test, y_proba, name,
            models_dir / f"{safe_name}_roc_curve.png",
        )
    plot_feature_importance(
        model,
        list(X_test.columns),
        name,
        models_dir / f"{safe_name}_feature_importance.png",
    )
    return metrics


def save_model(name: str, model: Any, models_dir: Path = MODELS_PATH) -> Path:
    """Pickle the trained model to disk."""
    safe_name = name.replace(" ", "_")
    path = models_dir / f"{safe_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model → {path}")
    return path


def _print_summary(results: Dict[str, Dict[str, float]]) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*70}")
    print(" MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':<25s}"
    metric_names = list(next(iter(results.values())).keys())
    for m in metric_names:
        header += f"{m:>12s}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:<25s}"
        for m in metric_names:
            row += f"{metrics.get(m, 0):>12.4f}"
        print(row)
    print(f"{'='*70}")


def model_training_service(
    data_path: Path = None,
    models_dir: Path = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """Execute the complete training pipeline and return results.

    Parameters
    ----------
    data_path : Path, optional
        Path to the ML-ready CSV. Defaults to PROCESSED_DATA_PATH / "ml_ready_dataset.csv".
    models_dir : Path, optional
        Directory to save models and plots. Defaults to MODELS_PATH.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping of model name → metrics dict.
    """
    if data_path is None:
        data_path = PROCESSED_DATA_PATH / "ml_ready_dataset.csv"
    if models_dir is None:
        models_dir = MODELS_PATH

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df, test_size, random_state)

    # Define baseline models
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        trained = train_model(name, model, X_train, y_train)
        metrics = evaluate(name, trained, X_test, y_test, models_dir)
        save_model(name, trained, models_dir)
        results[name] = metrics

    # Summary comparison
    _print_summary(results)
    return results


if __name__ == "__main__":
    model_training_service()
