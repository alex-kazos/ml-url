import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Services.inference_service import inference_service


def main() -> None:
    print("=" * 60)
    print("  Phishing URL Detection - Inference Pipeline")
    print("=" * 60)

    # label = 1 -> suspicious / phishing
    # label = 0 -> legitimate / safe
    url = input("\nEnter a URL to check: ").strip()

    if not url:
        print("No URL entered. Exiting.")
        return

    result = inference_service(url)

    print("\n--- Summary ---")
    print(f"  URL         : {result['url']}")
    print(f"  Probability : {result['probability']:.4f}  (phishing likelihood)")
    print(f"  Label       : {result['label']}  (1 = suspicious, 0 = legitimate)")
    print(f"  Verdict     : {result['verdict']}")


if __name__ == "__main__":
    main()
