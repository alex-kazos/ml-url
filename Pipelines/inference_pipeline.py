"""
Inference Pipeline
==================
Entry point for phishing-URL detection at serve time.

Usage
-----
Run from the project root directory:

    python -m Pipelines.inference_pipeline

or simply:

    python Pipelines/inference_pipeline.py

The user is prompted to enter a URL. The pipeline then:
  1. Preprocesses the URL (URL decomposition, character features, etc.)
  2. Applies feature engineering (entropy, path features, keywords, …)
  3. Loads the trained ``Random_Forest.pkl`` model
  4. Returns a probability that the URL is suspicious / phishing

Output convention
-----------------
  label = 1  →  suspicious / phishing
  label = 0  →  legitimate / safe
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Services.inference_service import inference_service


def main() -> None:
    print("=" * 60)
    print("  Phishing URL Detection — Inference Pipeline")
    print("=" * 60)

    url = input("\nEnter a URL to check: ").strip()

    if not url:
        print("No URL entered. Exiting.")
        return

    result = inference_service(url)

    # Surface the key numbers cleanly for the user
    print("\n--- Summary ---")
    print(f"  URL         : {result['url']}")
    print(f"  Probability : {result['probability']:.4f}  (phishing likelihood)")
    print(f"  Label       : {result['label']}  (1 = suspicious, 0 = legitimate)")
    print(f"  Verdict     : {result['verdict']}")


if __name__ == "__main__":
    main()
