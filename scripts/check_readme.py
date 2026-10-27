#!/usr/bin/env python3
"""
Validates that summary benchmark numbers reported in README.md
exactly equal the medians in results/latest.json.
Exits 0 on success, exits 1 on mismatch.
"""
import json
import sys
import os

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(root, "results", "latest.json")
    readme_path = os.path.join(root, "README.md")

    if not os.path.exists(results_path):
        print(f"[FAIL] Results file not found: {results_path}")
        sys.exit(1)

    if not os.path.exists(readme_path):
        print(f"[FAIL] README file not found: {readme_path}")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stats = data["stats"]

    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    missing = []

    # Check throughput medians across all batch sizes
    for bs in [32, 64, 128, 256, 512]:
        for fw, label in [("torch", "PyTorch"), ("mlx", "Apple MLX"), ("candle", "Candle")]:
            tput = round(stats[fw]["throughput"][str(bs)]["median"])
            formatted = f"{tput:,} img/s"
            if formatted not in readme:
                missing.append(f"Batch {bs} {label} throughput: {formatted}")

    # Check batch-1 latency percentiles
    for p, label in [("p50", "P50"), ("p95", "P95"), ("p99", "P99")]:
        for fw, flabel in [("torch", "PyTorch"), ("mlx", "Apple MLX"), ("candle", "Candle")]:
            val = stats[fw]["latency"][p]["median"]
            formatted = f"{val:.1f} μs"
            if formatted not in readme:
                missing.append(f"Latency {label} {flabel}: {formatted}")

    if missing:
        print(f"[FAIL] {len(missing)} metrics in results/latest.json were not found in README.md:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)

    print(f"[PASS] All 24 summary metrics in README.md match results/latest.json medians.")
    sys.exit(0)

if __name__ == "__main__":
    main()
