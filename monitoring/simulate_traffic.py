"""Send realistic traffic to the API so the dashboard and alerts have data.

    python monitoring/simulate_traffic.py --minutes 10            # normal day
    python monitoring/simulate_traffic.py --minutes 10 --attack   # phishing wave

The URL pool is the golden set plus variations typed with and without a scheme.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import requests

GOLDEN_SET = Path(__file__).resolve().parent.parent / "Models" / "golden_set.csv"


def load_pool():
    legit, phishing = [], []
    with open(GOLDEN_SET, newline="") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            url = row["url"]
            bare = url.split("://", 1)[-1]
            variants = {url, bare, f"https://{bare}"}
            (phishing if row["label"] == "1" else legit).extend(variants)
    return legit, phishing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--minutes", type=float, default=5)
    parser.add_argument("--rps", type=float, default=2.0, help="requests per second")
    parser.add_argument("--attack", action="store_true", help="60%% phishing instead of 10%%")
    args = parser.parse_args()

    legit, phishing = load_pool()
    phishing_share = 0.6 if args.attack else 0.1
    deadline = time.time() + args.minutes * 60
    sent = 0

    while time.time() < deadline:
        url = random.choice(phishing if random.random() < phishing_share else legit)
        try:
            requests.post(f"{args.api}/predict", json={"url": url}, timeout=10)
            sent += 1
        except requests.RequestException as exc:
            print(f"request failed: {exc}")
        time.sleep(1 / args.rps)

    print(f"Sent {sent} requests ({phishing_share:.0%} phishing).")


if __name__ == "__main__":
    main()
