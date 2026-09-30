"""Compare the challenger with the champion and export the winner for the API.

Usage:
    python Pipelines/promotion_pipeline.py            # compare, maybe promote, export
    python Pipelines/promotion_pipeline.py --export   # only re-export the current champion
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Services.model_registry import export_champion, promote_challenger  # noqa: E402


def run_promotion_pipeline(export_only: bool = False) -> dict:
    summary = {}
    if not export_only:
        summary = promote_challenger()
        if not summary["promoted"]:
            print("Champion unchanged; re-exporting it so Models/ matches the registry.")
    export_champion()
    return summary


if __name__ == "__main__":
    run_promotion_pipeline(export_only="--export" in sys.argv)
