"""
Master Benchmark Orchestrator: PyTorch (MPS) vs. Apple MLX vs. Hugging Face Candle (Rust).
Runs all 3 frameworks on identical CNN architectures, compares throughput and batch-1 latency
across multiple trials with rotated framework order, and outputs a publication-quality
comparative visualization with error bars and machine-readable telemetry.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

def get_thermal_state():
    try:
        res = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"pmset error: {e}"

def parse_json_result(result):
    data = {}
    for line in result.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            json_str = line.replace("JSON_RESULT:", "").strip()
            data = json.loads(json_str)
            break

    if not data:
        raise RuntimeError(
            "Benchmark process produced no JSON_RESULT line.\n--- stdout ---\n"
            f"{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    if "Metal init failed" in result.stderr:
        raise RuntimeError("Candle fell back to CPU; results would not be comparable:\n" + result.stderr)

    # Normalize throughput keys to integers
    data["throughput"] = {int(k): float(v) for k, v in data.get("throughput", {}).items()}
    return data

def run_framework_subprocess(cmd, cwd=None):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
    return parse_json_result(result)

def ensure_candle_binary():
    rust_dir = os.path.join(os.path.dirname(__file__), "rust_candle")
    binary_path = os.path.join(rust_dir, "target", "release", "rust_candle")
    if not os.path.exists(binary_path):
        cargo_bin = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
        if not os.path.exists(cargo_bin):
            raise RuntimeError("cargo not found on PATH or at ~/.cargo/bin; install Rust via rustup or `brew install rust`")
        print("[Candle] Compiling release binary with Metal support...")
        subprocess.run([cargo_bin, "build", "--release"], cwd=rust_dir, check=True)
    return binary_path

def main():
    parser = argparse.ArgumentParser(description="Master DL Shootout Orchestrator")
    parser.add_argument("--trials", type=int, default=5, help="Number of benchmark trials to run (default: 5)")
    parser.add_argument("--idle-seconds", type=float, default=2.0, help="Idle pause between framework runs (default: 2.0s)")
    parser.add_argument("--include-compiled", action="store_true", help="Include torch.compile column in benchmark")
    parser.add_argument("--output-plot", type=str, default="assets/shootout_results.png", help="Path to output plot PNG")
    parser.add_argument("--results-json", type=str, default="results/latest.json", help="Path to output results JSON")
    args = parser.parse_args()

    n_trials = args.trials
    batch_sizes = [32, 64, 128, 256, 512]
    candle_binary = ensure_candle_binary()

    print("=" * 80)
    print("APPLE SILICON DEEP LEARNING SHOOTOUT (Apple M4 Mac Mini)")
    print(f"PyTorch (MPS) vs. Apple MLX (GPU) vs. Hugging Face Candle (Rust Metal) — {n_trials} Trials")
    print("=" * 80)

    # 1. Telemetry & Environment Metadata
    thermal_before = get_thermal_state()

    # Read versions from python modules if possible
    try:
        import torch as _t
        torch_ver = _t.__version__
    except ImportError:
        torch_ver = "unknown"

    try:
        import mlx.core as _mx
        mlx_ver = _mx.__version__
    except ImportError:
        mlx_ver = "unknown"

    candle_ver = "0.11.0"

    framework_defs = [
        ("torch", "PyTorch (Apple MPS)", [sys.executable, "python/torch_bench.py", "--json", "--floor"]),
        ("mlx", "Apple MLX (Native GPU)", [sys.executable, "python/mlx_bench.py", "--json", "--floor"]),
        ("candle", "Candle (Rust Metal)", [candle_binary]),
    ]
    if args.include_compiled:
        framework_defs.append(
            ("torch_compile", "PyTorch (MPS, torch.compile)", [sys.executable, "python/torch_bench.py", "--json", "--compile"])
        )

    framework_keys = [f[0] for f in framework_defs]
    runs = {k: [] for k in framework_keys}

    # 2. Multi-Trial Loop with Rotated Order
    for trial in range(n_trials):
        # Rotate framework order across trials
        shift = trial % len(framework_defs)
        trial_frameworks = framework_defs[shift:] + framework_defs[:shift]
        print(f"\n--- TRIAL {trial + 1}/{n_trials} (Execution Order: {', '.join([f[1] for f in trial_frameworks])}) ---")

        for key, name, cmd in trial_frameworks:
            print(f"  Running {name}...")
            data = run_framework_subprocess(cmd)
            runs[key].append(data)
            if args.idle_seconds > 0:
                time.sleep(args.idle_seconds)

    thermal_after = get_thermal_state()

    # 3. Aggregate Statistics (Medians, Mins, Maxs)
    stats = {}
    for key in framework_keys:
        f_runs = runs[key]
        tput_stats = {}
        for bs in batch_sizes:
            vals = [r["throughput"][bs] for r in f_runs]
            tput_stats[bs] = {
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "vals": vals,
            }

        lat_stats = {}
        for p in ["p50", "p95", "p99"]:
            vals = [r["latency_us"][p] for r in f_runs]
            lat_stats[p] = {
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "vals": vals,
            }

        floors = [r["dispatch_floor_us"] for r in f_runs if "dispatch_floor_us" in r]
        floor_median = float(np.median(floors)) if floors else None

        stats[key] = {
            "name": f_runs[0].get("engine", key),
            "throughput": tput_stats,
            "latency": lat_stats,
            "dispatch_floor_us": floor_median,
        }

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 80)
    print(f"FINAL COMPARISON TABLE (Medians across {n_trials} trials with min–max spread)")
    print("=" * 80)
    header_cols = [stats[k]["name"] for k in framework_keys]
    header_line = f"{'Metric':<28} | " + " | ".join([f"{col:<22}" for col in header_cols])
    print(header_line)
    print("-" * len(header_line))

    for bs in batch_sizes:
        row_str = f"Batch {bs:<3} Throughput (img/s)  | "
        col_vals = []
        for k in framework_keys:
            m = stats[k]["throughput"][bs]["median"]
            mn = stats[k]["throughput"][bs]["min"]
            mx = stats[k]["throughput"][bs]["max"]
            col_vals.append(f"{m:,.0f} ({mn:,.0f}–{mx:,.0f})")
        row_str += " | ".join([f"{val:<22}" for val in col_vals])
        print(row_str)

    print("-" * len(header_line))
    for p_label, p_key in [("P50 (Median)", "p50"), ("P95 (Tail)", "p95"), ("P99 (Max)", "p99")]:
        row_str = f"Batch-1 Latency {p_label:<14} | "
        col_vals = []
        for k in framework_keys:
            m = stats[k]["latency"][p_key]["median"]
            mn = stats[k]["latency"][p_key]["min"]
            mx = stats[k]["latency"][p_key]["max"]
            col_vals.append(f"{m:.1f} ({mn:.1f}–{mx:.1f}) μs")
        row_str += " | ".join([f"{val:<22}" for val in col_vals])
        print(row_str)

    # Dispatch floor row
    row_str = f"{'Python Dispatch Floor P50':<28} | "
    floor_vals = []
    for k in framework_keys:
        f_val = stats[k]["dispatch_floor_us"]
        if f_val is not None:
            floor_vals.append(f"{f_val:.2f} μs")
        else:
            floor_vals.append("N/A (Pure Binary)")
    row_str += " | ".join([f"{val:<22}" for val in floor_vals])
    print(row_str)
    print("=" * len(header_line))

    # 5. Export Results JSON
    run_meta = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch_ver,
        "mlx": mlx_ver,
        "candle_core": candle_ver,
        "batch_sizes": batch_sizes,
        "n_trials": n_trials,
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
    }
    payload = {
        "meta": run_meta,
        "stats": stats,
        "raw_runs": runs,
    }
    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[✓] Machine-readable results saved to: {os.path.abspath(args.results_json)}")

    # 6. Generate Publication-Quality Visualizations with Error Bars
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    x_indices = np.arange(len(batch_sizes))
    colors = {
        "torch": "#EE4C2C",
        "mlx": "#0071E3",
        "candle": "#FFA500",
        "torch_compile": "#8E44AD",
    }
    markers = {
        "torch": "o-",
        "mlx": "s-",
        "candle": "^-",
        "torch_compile": "d--",
    }

    # Panel 1: Throughput Curves across Batch Sizes with Error Bars
    for key in framework_keys:
        medians = [stats[key]["throughput"][b]["median"] for b in batch_sizes]
        mins = [stats[key]["throughput"][b]["min"] for b in batch_sizes]
        maxs = [stats[key]["throughput"][b]["max"] for b in batch_sizes]
        yerr = [
            np.maximum(0, np.array(medians) - np.array(mins)),
            np.maximum(0, np.array(maxs) - np.array(medians)),
        ]
        axes[0].errorbar(
            x_indices,
            medians,
            yerr=yerr,
            fmt=markers.get(key, "o-"),
            label=stats[key]["name"],
            color=colors.get(key, "#333333"),
            lw=2.5,
            ms=7,
            capsize=4,
            capthick=1.5,
        )

    axes[0].set_title("Throughput Scaling vs. Batch Size (Images / Second)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Batch Size", fontsize=11)
    axes[0].set_ylabel("Throughput (Images / Sec)", fontsize=11)
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels([str(b) for b in batch_sizes])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Panel 2: Batch-1 Latency Bars with Error Bars
    bar_x = np.arange(len(framework_keys))
    width = 0.35

    p50_medians = [stats[k]["latency"]["p50"]["median"] for k in framework_keys]
    p50_mins = [stats[k]["latency"]["p50"]["min"] for k in framework_keys]
    p50_maxs = [stats[k]["latency"]["p50"]["max"] for k in framework_keys]
    p50_yerr = [
        np.maximum(0, np.array(p50_medians) - np.array(p50_mins)),
        np.maximum(0, np.array(p50_maxs) - np.array(p50_medians)),
    ]

    p95_medians = [stats[k]["latency"]["p95"]["median"] for k in framework_keys]
    p95_mins = [stats[k]["latency"]["p95"]["min"] for k in framework_keys]
    p95_maxs = [stats[k]["latency"]["p95"]["max"] for k in framework_keys]
    p95_yerr = [
        np.maximum(0, np.array(p95_medians) - np.array(p95_mins)),
        np.maximum(0, np.array(p95_maxs) - np.array(p95_medians)),
    ]

    b1 = axes[1].bar(
        bar_x - width / 2,
        p50_medians,
        width,
        yerr=p50_yerr,
        capsize=3,
        label="P50 (Median Latency)",
        color="#34C759",
    )
    b2 = axes[1].bar(
        bar_x + width / 2,
        p95_medians,
        width,
        yerr=p95_yerr,
        capsize=3,
        label="P95 (Tail Latency)",
        color="#FF9500",
    )

    axes[1].set_title("Single-Image Inference Latency (Batch Size = 1)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Latency (Microseconds - Lower is Better)", fontsize=11)
    axes[1].set_xticks(bar_x)
    axes[1].set_xticklabels([stats[k]["name"] for k in framework_keys], fontweight="bold", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=9)

    for bar, val in zip(b1, p50_medians):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 15, f"{val:.0f}μs", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(b2, p95_medians):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 15, f"{val:.0f}μs", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metadata Stamp in Footer
    stamp_text = (
        f"{run_meta['timestamp_utc']} | macOS {run_meta['macos']} ({run_meta['machine']}) | "
        f"torch {run_meta['torch']} | mlx {run_meta['mlx']} | candle {run_meta['candle_core']} | "
        f"n={n_trials} trials (median ± min-max)"
    )
    fig.text(0.02, 0.015, stamp_text, fontsize=7, color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(args.output_plot, dpi=160)
    print(f"[✓] Shootout Plot with error bars saved to: {os.path.abspath(args.output_plot)}")

if __name__ == "__main__":
    main()
