"""Retrain on fresh data, then run the champion/challenger promotion.

Schedule this (e.g. weekly) or trigger it from a drift alert. It never deploys
by itself: review the promotion result, then commit Models/champion.pkl.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Pipelines.training_pipeline import run_training_pipeline  # noqa: E402
from Pipelines.promotion_pipeline import run_promotion_pipeline  # noqa: E402


if __name__ == "__main__":
    run_training_pipeline()
    run_promotion_pipeline()
