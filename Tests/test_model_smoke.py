"""Smoke test for the model file the API actually serves.

Unit tests prove the code is right; this proves the *artifact* is right. It runs
the golden set through the same prediction path as /predict, so a model trained
on stale or mislabelled data fails here before it reaches users.
"""
import os

import pytest

from Services.inference_service import (
    _load_model,
    _preprocess_url_for_inference,
    predict_with_bundle,
)
from Services.model_registry import bundle_predictor, golden_set_check
from Utilities.config import CHAMPION_FILE_STEM, GOLDEN_SET_MIN_ACCURACY, MODELS_PATH

SERVED = [os.getenv("API_MODEL_NAME", CHAMPION_FILE_STEM)]
if (MODELS_PATH / f"{CHAMPION_FILE_STEM}.pkl").exists() and CHAMPION_FILE_STEM not in SERVED:
    SERVED.append(CHAMPION_FILE_STEM)


def _load_served(model_name):
    if not (MODELS_PATH / f"{model_name}.pkl").exists():
        pytest.skip(f"Models/{model_name}.pkl not found")
    return _load_model(model_name)


@pytest.mark.parametrize("model_name", SERVED)
def test_served_model_passes_golden_set(model_name):
    model, metadata = _load_served(model_name)
    accuracy, failures = golden_set_check(bundle_predictor(model, metadata))

    details = "\n".join(
        f"  expected {f['expected']}, got {f['predicted']}: {f['url'][:90]}" for f in failures
    )
    assert accuracy >= GOLDEN_SET_MIN_ACCURACY, (
        f"Models/{model_name}.pkl scores {accuracy:.2f} on the golden set "
        f"(minimum {GOLDEN_SET_MIN_ACCURACY:.2f}). Wrong answers:\n{details}"
    )


@pytest.mark.parametrize("model_name", SERVED)
def test_inference_computes_every_feature_the_model_expects(model_name):
    # predict_with_bundle reindexes to feature_names with fill_value=0, so a
    # feature that inference stopped computing would silently become 0.
    _, metadata = _load_served(model_name)

    computed = set(_preprocess_url_for_inference("example.com/login").columns)

    assert set(metadata["feature_names"]) <= computed


# Uses the bundle's own reference list (top 100k domains), so these check the
# lookalike rule as users get it, not a hand-made list.
@pytest.mark.parametrize("model_name", SERVED)
@pytest.mark.parametrize("url", ["substack.com", "sfnmjournal.com", "mlflow.org", "vercel.com"])
def test_lookalike_rule_leaves_legitimate_sites_alone(model_name, url):
    model, metadata = _load_served(model_name)

    result, _ = predict_with_bundle(url, model, metadata)

    assert "risk_signals" not in result, result["risk_signals"]


@pytest.mark.parametrize("model_name", SERVED)
@pytest.mark.parametrize("url", ["linkedln.com", "faceb00k.com", "paypa1.com"])
def test_lookalike_rule_still_catches_typosquats(model_name, url):
    model, metadata = _load_served(model_name)

    result, _ = predict_with_bundle(url, model, metadata)

    assert result.get("risk_signals"), f"{url} raised no lookalike signal"
    assert result["label"] == 1
