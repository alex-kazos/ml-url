import csv
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple
from urllib.parse import urlparse

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
from Utilities.config import KAGGLE_TOP_SEARCHES_FILE_PATH, MODELS_PATH


COLUMNS_TO_DROP = [
    "URL",
    "Domain",
    "TLD",
    "Label",
    "TLDLegitimateProb",
    "label_binary",
]

# chose this for performance
DEFAULT_MODEL_NAME = "XGBoost"

LEETSPEAK_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
    }
)

COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac",
    "co",
    "com",
    "edu",
    "gov",
    "net",
    "org",
}

DEFAULT_REFERENCE_DOMAIN_LIMIT = 100_000
MIN_LOOKALIKE_LABEL_LENGTH = 5


class ReferenceDomainIndex(NamedTuple):
    labels: dict[str, str]
    labels_by_length: dict[int, dict[str, str]]
    labels_by_bigram: dict[str, frozenset[str]]


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


@lru_cache(maxsize=8)
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


def _hostname_from_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return (parsed.hostname or "").lower().strip(".")


def _registered_domain_label(hostname: str) -> str:
    labels = [label for label in hostname.split(".") if label]
    if len(labels) < 2:
        return labels[0] if labels else ""

    if (
        len(labels) >= 3
        and labels[-2] in COMMON_SECOND_LEVEL_SUFFIXES
        and len(labels[-1]) == 2
    ):
        return labels[-3]

    return labels[-2]


def _registered_domain(hostname: str) -> str:
    labels = [label for label in hostname.split(".") if label]
    if len(labels) < 2:
        return labels[0] if labels else ""

    if (
        len(labels) >= 3
        and labels[-2] in COMMON_SECOND_LEVEL_SUFFIXES
        and len(labels[-1]) == 2
    ):
        return ".".join(labels[-3:])

    return ".".join(labels[-2:])


def _compact_hostname_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _bigrams(text: str) -> frozenset[str]:
    if len(text) < 2:
        return frozenset()
    return frozenset(text[i : i + 2] for i in range(len(text) - 1))


def _bigram_similarity(left: str, right: str) -> float:
    left_bigrams = _bigrams(left)
    right_bigrams = _bigrams(right)
    if not left_bigrams or not right_bigrams:
        return 0.0

    overlap = len(left_bigrams & right_bigrams)
    return (2 * overlap) / (len(left_bigrams) + len(right_bigrams))


def _domain_from_reference_row(row: list[str]) -> str:
    for cell in row:
        value = cell.strip().lower()
        if "." in value and not value.replace(".", "").isdigit():
            hostname = _hostname_from_url(value)
            return _registered_domain(hostname)
    return ""


@lru_cache(maxsize=4)
def _load_reference_domains(
    source_path: str = str(KAGGLE_TOP_SEARCHES_FILE_PATH),
    limit: int = DEFAULT_REFERENCE_DOMAIN_LIMIT,
) -> tuple[str, ...]:
    path = Path(source_path)
    if not path.exists():
        return ()

    domains: list[str] = []
    seen: set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                domain = _domain_from_reference_row(row)
                if not domain or domain in seen:
                    continue

                seen.add(domain)
                domains.append(domain)
                if len(domains) >= limit:
                    break
    except OSError:
        return ()

    return tuple(domains)


@lru_cache(maxsize=16)
def _reference_domain_index(reference_domains: tuple[str, ...]) -> ReferenceDomainIndex:
    labels: dict[str, str] = {}
    for domain in reference_domains:
        hostname = _hostname_from_url(domain)
        registered_domain = _registered_domain(hostname)
        label = _compact_hostname_label(_registered_domain_label(registered_domain))
        if len(label) >= MIN_LOOKALIKE_LABEL_LENGTH and label not in labels:
            labels[label] = registered_domain

    labels_by_length: dict[int, dict[str, str]] = {}
    for label, domain in labels.items():
        labels_by_length.setdefault(len(label), {})[label] = domain

    bigram_labels: dict[str, set[str]] = {}
    for label in labels:
        for bigram in _bigrams(label):
            bigram_labels.setdefault(bigram, set()).add(label)
    labels_by_bigram = {
        bigram: frozenset(bigram_labels) for bigram, bigram_labels in bigram_labels.items()
    }

    return ReferenceDomainIndex(
        labels=labels,
        labels_by_length=labels_by_length,
        labels_by_bigram=labels_by_bigram,
    )


def _reference_domain_candidates(
    normalized_label: str,
    index: ReferenceDomainIndex,
) -> dict[str, str]:
    min_length = len(normalized_label) - 2
    max_length = len(normalized_label) + 2
    matching_labels: set[str] = set()
    for bigram in _bigrams(normalized_label):
        matching_labels.update(index.labels_by_bigram.get(bigram, ()))

    candidates = {}
    for label in matching_labels:
        if not min_length <= len(label) <= max_length:
            continue
        if _bigram_similarity(normalized_label, label) < 0.55:
            continue
        candidates[label] = index.labels[label]
    return candidates


def _reference_index_from_metadata(metadata: Dict[str, Any]) -> ReferenceDomainIndex:
    index = metadata.get("reference_domain_index")
    if isinstance(index, ReferenceDomainIndex):
        return index

    references = metadata.get("reference_domains")
    if references is None:
        references = _load_reference_domains()

    return _reference_domain_index(tuple(references))


def _edit_distance_at_most(left: str, right: str, max_edits: int) -> bool:
    if abs(len(left) - len(right)) > max_edits:
        return False

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = i
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            value = min(insertion, deletion, substitution)
            current.append(value)
            row_min = min(row_min, value)

        if row_min > max_edits:
            return False
        previous = current

    return previous[-1] <= max_edits


def _domain_impersonation_signals(
    url: str,
    reference_domains: Optional[Iterable[str]] = None,
    reference_index: Optional[ReferenceDomainIndex] = None,
) -> list[str]:
    hostname = _hostname_from_url(url)
    root_label = _compact_hostname_label(_registered_domain_label(hostname))
    if len(root_label) < MIN_LOOKALIKE_LABEL_LENGTH:
        return []

    normalized_root = root_label.translate(LEETSPEAK_TRANSLATION)
    if reference_index is None:
        references = reference_domains
        if references is None:
            references = _load_reference_domains()
        reference_index = _reference_domain_index(tuple(references))

    signals = []
    exact_domain = reference_index.labels.get(normalized_root)
    if exact_domain and root_label != normalized_root:
        signals.append(f"domain-lookalike:{root_label}->{exact_domain}")
        return signals

    if root_label == normalized_root and exact_domain:
        return []

    candidates = _reference_domain_candidates(normalized_root, reference_index)
    for reference_label, reference_domain in candidates.items():
        if reference_label == normalized_root:
            continue

        if _edit_distance_at_most(normalized_root, reference_label, max_edits=2):
            signals.append(f"domain-typosquat:{root_label}->{reference_domain}")

    return signals


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
    risk_signals = _domain_impersonation_signals(
        url,
        reference_index=_reference_index_from_metadata(metadata),
    )
    if risk_signals and label == 0:
        label = 1
        phishing_probability = max(phishing_probability, 0.85)

    verdict = "[!] SUSPICIOUS / PHISHING" if label == 1 else "[OK] LEGITIMATE"

    result = {
        "url": url,
        "label": label,
        "probability": round(phishing_probability, 4),
        "verdict": verdict,
    }
    if risk_signals:
        result["risk_signals"] = risk_signals

    print("\n  Result:")
    print(f"    URL         : {url}")
    print(f"    Label       : {label}  (1 = suspicious, 0 = legitimate)")
    print(f"    Probability : {phishing_probability:.4f}  (probability of phishing)")
    if risk_signals:
        print(f"    Risk signals: {', '.join(risk_signals)}")
    print(f"    Verdict     : {verdict}")
    print(f"{'='*60}\n")

    return result
