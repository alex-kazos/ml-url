
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Import utility functions used during training
from Utilities.Services.preprocess_data_utils import (
    _add_basic_url_parts,
    _add_char_count_and_ratio_features,
    _add_obfuscation_features,
    _add_keyword_flags,
    _add_char_continuation_feature,
    _add_url_char_prob_and_similarity,
)
from Utilities.Services.feature_engineering_utils import add_all_new_features

# Import configuration
from Utilities.config import MODELS_PATH


# Columns to be dropped
COLUMNS_TO_DROP = [
    "URL",
    "Domain",
    "TLD",
    "Label",
    "TLDLegitimateProb",
    "label_binary",
]

# Default model file
DEFAULT_MODEL_NAME = "Random_Forest"



def _preprocess_url_for_inference(url: str) -> pd.DataFrame:
    """Apply the same preprocessing steps as the training pipeline.

    This is a *inference-safe* variant of ``preprocess_phishing`` — it skips
    the steps that require a labelled dataset (``label_binary``,
    ``TLDLegitimateProb``) because those columns are dropped before the model
    sees any data anyway.

    Parameters
    ----------
    url : str
        The raw URL to analyse.

    Returns
    -------
    pd.DataFrame
        A single-row dataframe with all URL-derived features populated.
    """
    # Start with a minimal dataframe containing only the URL column,
    # naming it "URL" to match the column name expected by the utils.
    df = pd.DataFrame({"URL": [url]})

    # -- Stage 1: basic URL decomposition (length, domain, TLD, etc.) ------
    df, url_s, lower_url = _add_basic_url_parts(df)

    # -- Stage 2: character count / ratio features -------------------------
    df = _add_char_count_and_ratio_features(df, url_s)

    # -- Stage 3: obfuscation signals -------------------------------------
    df = _add_obfuscation_features(df, url_s)

    # -- Stage 4: keyword flags -------------------------------------------
    df = _add_keyword_flags(df, lower_url)

    # -- Stage 5: character continuation ----------------------------------
    df = _add_char_continuation_feature(df, url_s)

    # -- Stage 6: URL character probability model -------------------------
    # For a single URL the "corpus" is just that URL itself; the resulting
    # probability is still a valid structural signal.
    df = _add_url_char_prob_and_similarity(df, url_s)

    # -- Stage 7: additional feature engineering (entropy, path, etc.) -----
    df = add_all_new_features(df)

    # -- Tidy up: replace inf/NaN with 0 (same as training pipeline) -------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def _drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all columns that were excluded during model training."""
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def _load_model(model_name: str = DEFAULT_MODEL_NAME, models_dir: Path = MODELS_PATH):
    """Load a pickled model from the Models directory.

    Parameters
    ----------
    model_name : str
        Base name of the model file (without ``.pkl`` extension).
    models_dir : Path
        Directory where model ``.pkl`` files are stored.

    Returns
    -------
    Fitted scikit-learn / compatible model object.

    Raises
    ------
    FileNotFoundError
        If the ``.pkl`` file does not exist at the expected path.
    """
    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run the training pipeline first to generate the model."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded model from: {model_path}")
    return model



def inference_service(
    url: str,
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: Path = MODELS_PATH,
) -> Dict[str, Any]:
    """Run end-to-end inference on a single URL.

    Steps
    -----
    1. Preprocess the URL (URL decomposition, character features, etc.).
    2. Apply additional feature engineering (entropy, path features, etc.).
    3. Drop non-feature columns to match training schema.
    4. Load the trained model from disk.
    5. Predict label and phishing probability.

    Parameters
    ----------
    url : str
        The URL to classify.
    model_name : str, optional
        Name of the model to load (defaults to ``"Random_Forest"``).
    models_dir : Path, optional
        Directory containing the pickled model files.

    Returns
    -------
    dict with keys:
        - ``url``         (str)   — the input URL
        - ``label``       (int)   — 1 = suspicious/phishing, 0 = legitimate
        - ``probability`` (float) — model probability that the URL is phishing
        - ``verdict``     (str)   — human-readable verdict
    """
    print(f"\n{'='*60}")
    print(f"  Inference Service — URL: {url}")
    print(f"{'='*60}")

    # Step 1 & 2: Preprocessing + feature engineering
    print("\n[1/3] Preprocessing & feature engineering …")
    feature_df = _preprocess_url_for_inference(url)
    print(f"       Generated {feature_df.shape[1]} raw feature columns.")

    # Step 3: Align columns with training schema
    X = _drop_non_feature_columns(feature_df)

    # Reindex to *exactly* match the column order the model was trained with.
    # If the model has `feature_names_in_`, use it; otherwise trust column order.
    print(f"       Using {X.shape[1]} features for prediction.")

    # Step 4: Load model
    print(f"\n[2/3] Loading model '{model_name}' …")
    model = _load_model(model_name, models_dir)

    # Step 5: Predict
    print("\n[3/3] Predicting …")

    # Align to exact model feature order (critical when model has feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    raw_label = int(model.predict(X)[0])

    phishing_probability = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # proba[0] = P(label_binary == 0) = P(phishing) — directly use this
        phishing_probability = float(proba[0])
    elif hasattr(model, "decision_function"):
        # Fallback for models without predict_proba (e.g. LinearSVC)
        # decision_function > 0 → predicts class 1 (legitimate)
        raw_score = float(model.decision_function(X)[0])
        legit_prob = float(1 / (1 + np.exp(-raw_score)))  # sigmoid → P(legitimate)
        phishing_probability = 1.0 - legit_prob

    # Invert label so 1 = suspicious matches the user-facing convention
    is_suspicious = 1 if raw_label == 0 else 0

    verdict = "[!] SUSPICIOUS / PHISHING" if is_suspicious else "[OK] LEGITIMATE"

    result = {
        "url": url,
        "label": is_suspicious,
        "probability": round(phishing_probability, 4),
        "verdict": verdict,
    }

    print(f"\n  Result:")
    print(f"    URL         : {url}")
    print(f"    Label       : {is_suspicious}  (1 = suspicious, 0 = legitimate)")
    print(
        f"    Probability : {phishing_probability:.4f}  (probability of being phishing)"
    )
    print(f"    Verdict     : {verdict}")
    print(f"{'='*60}\n")

    return result
