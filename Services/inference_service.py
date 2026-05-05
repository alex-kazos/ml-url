import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from Utilities.Services.preprocess_data_utils import (
    URLCharacterProbabilityModel,
    _add_basic_url_parts,
    _add_char_continuation_feature,
    _add_char_count_and_ratio_features,
    _add_keyword_flags,
    _add_obfuscation_features,
    _add_url_char_prob_and_similarity,
    apply_url_char_probability_model,
)
from Utilities.Services.feature_engineering_utils import add_all_new_features
from Utilities.config import MODELS_PATH


COLUMNS_TO_DROP = [
    "URL",
    "Domain",
    "TLD",
    "Label",
    "TLDLegitimateProb",
    "label_binary",
]

DEFAULT_MODEL_NAME = "Random_Forest"


def _preprocess_url_for_inference(
    url: str,
    char_model: Optional[URLCharacterProbabilityModel] = None,
) -> pd.DataFrame:
    """Apply the same URL feature generation used by the training pipeline."""
    df = pd.DataFrame({"URL": [url]})

    df, url_s, lower_url = _add_basic_url_parts(df)
    df = _add_char_count_and_ratio_features(df, url_s)
    df = _add_obfuscation_features(df, url_s)
    df = _add_keyword_flags(df, lower_url)
    df = _add_char_continuation_feature(df, url_s)

    if char_model is not None:
        df = apply_url_char_probability_model(df, url_s, char_model)
    else:
        # Backward-compatible fallback for legacy model pickles.
        df = _add_url_char_prob_and_similarity(df, url_s)

    df = add_all_new_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def _drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all columns that were excluded during model training."""
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def _load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: Path = MODELS_PATH,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a model artifact from the Models directory.

    New artifacts are dictionaries containing the fitted model plus metadata.
    Raw estimator pickles are still accepted so older artifacts do not crash.
    """
    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run the training pipeline first to generate the model."
        )

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        metadata = {k: v for k, v in artifact.items() if k != "model"}
    else:
        model = artifact
        metadata = {}

    print(f"  Loaded model from: {model_path}")
    return model, metadata


def _phishing_probability(model: Any, X: pd.DataFrame) -> float:
    """Return P(label_binary == 1), where 1 means phishing."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", []))
        phishing_index = classes.index(1) if 1 in classes else min(1, len(proba) - 1)
        return float(proba[phishing_index])

    if hasattr(model, "decision_function"):
        raw_score = float(model.decision_function(X)[0])
        return float(1 / (1 + np.exp(-raw_score)))

    return 0.0


def inference_service(
    url: str,
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: Path = MODELS_PATH,
) -> Dict[str, Any]:
    """Run end-to-end inference on a single URL.

    The public convention is consistent with training:
    1 = suspicious/phishing, 0 = legitimate/safe.
    """
    print(f"\n{'='*60}")
    print(f"  Inference Service - URL: {url}")
    print(f"{'='*60}")

    print(f"\n[1/3] Loading model '{model_name}' ...")
    model, metadata = _load_model(model_name, models_dir)

    print("\n[2/3] Preprocessing & feature engineering ...")
    feature_df = _preprocess_url_for_inference(
        url,
        char_model=metadata.get("url_char_model"),
    )
    print(f"       Generated {feature_df.shape[1]} raw feature columns.")

    X = _drop_non_feature_columns(feature_df)
    feature_names = metadata.get("feature_names")
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)
    elif hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    print(f"       Using {X.shape[1]} features for prediction.")

    print("\n[3/3] Predicting ...")
    label = int(model.predict(X)[0])
    phishing_probability = _phishing_probability(model, X)
    verdict = "[!] SUSPICIOUS / PHISHING" if label == 1 else "[OK] LEGITIMATE"

    result = {
        "url": url,
        "label": label,
        "probability": round(phishing_probability, 4),
        "verdict": verdict,
    }

    print("\n  Result:")
    print(f"    URL         : {url}")
    print(f"    Label       : {label}  (1 = suspicious, 0 = legitimate)")
    print(f"    Probability : {phishing_probability:.4f}  (probability of phishing)")
    print(f"    Verdict     : {verdict}")
    print(f"{'='*60}\n")

    return result
