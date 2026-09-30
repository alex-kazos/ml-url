"""Smoke test for the model file the API actually serves.

Unit tests prove the code is right; this proves the *artifact* is right. It runs
the golden set through the same prediction path as /predict, so a model trained
on stale or mislabelled data fails here before it reaches users.
"""
import os

import pytest

from Services.inference_service import _load_model
from Services.model_registry import bundle_predictor, golden_set_check
from Utilities.config import CHAMPION_FILE_STEM, GOLDEN_SET_MIN_ACCURACY, MODELS_PATH

SERVED = [os.getenv("API_MODEL_NAME", "XGBoost")]
if (MODELS_PATH / f"{CHAMPION_FILE_STEM}.pkl").exists() and CHAMPION_FILE_STEM not in SERVED:
    SERVED.append(CHAMPION_FILE_STEM)


@pytest.mark.parametrize("model_name", SERVED)
def test_served_model_passes_golden_set(model_name):
    if not (MODELS_PATH / f"{model_name}.pkl").exists():
        pytest.skip(f"Models/{model_name}.pkl not found")

    model, metadata = _load_model(model_name)
    accuracy, failures = golden_set_check(bundle_predictor(model, metadata))

    details = "\n".join(
        f"  expected {f['expected']}, got {f['predicted']}: {f['url'][:90]}" for f in failures
    )
    assert accuracy >= GOLDEN_SET_MIN_ACCURACY, (
        f"Models/{model_name}.pkl scores {accuracy:.2f} on the golden set "
        f"(minimum {GOLDEN_SET_MIN_ACCURACY:.2f}). Wrong answers:\n{details}"
    )
