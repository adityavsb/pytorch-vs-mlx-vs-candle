"""
Master Benchmark Orchestrator: PyTorch (MPS) vs. Apple MLX vs. Hugging Face Candle (Rust).
Runs all 3 frameworks on identical CNN architectures, compares throughput and batch-1 latency,
and outputs a publication-quality comparative visualization.
"""
import os
import sys
import json
import shutil
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
from torch_bench import benchmark_torch_throughput, benchmark_torch_latency_batch1
from mlx_bench import benchmark_mlx_throughput, benchmark_mlx_latency_batch1

def run_candle_rust():
    cargo_bin = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo_bin):
        raise RuntimeError("cargo not found on PATH or at ~/.cargo/bin; install Rust via rustup or `brew install rust`")
    rust_dir = os.path.join(os.path.dirname(__file__), "rust_candle")
    binary_path = os.path.join(rust_dir, "target", "release", "rust_candle")

    if not os.path.exists(binary_path):
        print("[Candle] Compiling release binary with Metal support...")
        subprocess.run([cargo_bin, "build", "--release"], cwd=rust_dir, check=True)

    print("[Candle] Running Hugging Face Candle (Rust / Metal)...")
    result = subprocess.run([binary_path], capture_output=True, text=True, check=True)
    
    # Parse JSON output
    data = {}
    for line in result.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            json_str = line.replace("JSON_RESULT:", "").strip()
            data = json.loads(json_str)
            break
            
    if not data:
        raise RuntimeError(
            "Candle binary produced no JSON_RESULT line.\n--- stdout ---\n"
            f"{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    if "Metal init failed" in result.stderr:
        raise RuntimeError("Candle fell back to CPU; results would not be comparable:\n" + result.stderr)

    # Convert throughput keys to integers
    throughput = {int(k): float(v) for k, v in data.get("throughput", {}).items()}
    latency = data.get("latency_us", {"p50": 0.0, "p95": 0.0, "p99": 0.0})
    return throughput, latency

def main(output_plot="assets/shootout_results.png"):
    print("=" * 80)
    print("APPLE SILICON DEEP LEARNING SHOOTOUT (Apple M4 Mac Mini)")
    print("PyTorch (MPS) vs. Apple MLX (GPU) vs. Hugging Face Candle (Rust Metal)")
    print("=" * 80)

    batch_sizes = [32, 64, 128, 256, 512]

    # 1. Run PyTorch
    print("\n>>> 1. Benchmarking PyTorch (MPS) <<<")
    torch_tput = benchmark_torch_throughput(batch_sizes=batch_sizes)
    torch_lat = benchmark_torch_latency_batch1()

    # 2. Run Apple MLX
    print("\n>>> 2. Benchmarking Apple MLX (GPU) <<<")
    mlx_tput = benchmark_mlx_throughput(batch_sizes=batch_sizes)
    mlx_lat = benchmark_mlx_latency_batch1()

    # 3. Run Hugging Face Candle (Rust)
    print("\n>>> 3. Benchmarking Hugging Face Candle (Rust Metal) <<<")
    candle_tput, candle_lat = run_candle_rust()

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 80)
    print("FINAL 3-WAY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<28} | {'PyTorch (MPS)':<16} | {'Apple MLX':<16} | {'Candle (Rust)':<16}")
    print("-" * 80)
    for bs in batch_sizes:
        t_val = torch_tput[bs]
        m_val = mlx_tput[bs]
        c_val = candle_tput[bs]
        print(f"Batch {bs:<3} Throughput (img/s)  | {t_val:<16.0f} | {m_val:<16.0f} | {c_val:<16.0f}")

    t_p50 = f"{torch_lat['p50']:.1f} μs"
    m_p50 = f"{mlx_lat['p50']:.1f} μs"
    c_p50 = f"{candle_lat['p50']:.1f} μs"

    t_p95 = f"{torch_lat['p95']:.1f} μs"
    m_p95 = f"{mlx_lat['p95']:.1f} μs"
    c_p95 = f"{candle_lat['p95']:.1f} μs"

    t_p99 = f"{torch_lat['p99']:.1f} μs"
    m_p99 = f"{mlx_lat['p99']:.1f} μs"
    c_p99 = f"{candle_lat['p99']:.1f} μs"

    print("-" * 80)
    print(f"{'Batch-1 Latency P50':<28} | {t_p50:<16} | {m_p50:<16} | {c_p50:<16}")
    print(f"{'Batch-1 Latency P95':<28} | {t_p95:<16} | {m_p95:<16} | {c_p95:<16}")
    print(f"{'Batch-1 Latency P99':<28} | {t_p99:<16} | {m_p99:<16} | {c_p99:<16}")
    print("=" * 80)

    # 5. Generate Publication-Quality Visualizations
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    # Panel 1: Throughput Curves across Batch Sizes
    x_indices = np.arange(len(batch_sizes))
    axes[0].plot(x_indices, [torch_tput[b] for b in batch_sizes], "o-", label="PyTorch (Apple MPS)", color="#EE4C2C", lw=2.5, ms=8)
    axes[0].plot(x_indices, [mlx_tput[b] for b in batch_sizes], "s-", label="Apple MLX (Native GPU)", color="#0071E3", lw=2.5, ms=8)
    axes[0].plot(x_indices, [candle_tput[b] for b in batch_sizes], "^-", label="Candle (Rust Metal)", color="#FFA500", lw=2.5, ms=8)
    axes[0].set_title("Throughput Scaling vs. Batch Size (Images / Second)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Batch Size", fontsize=11)
    axes[0].set_ylabel("Throughput (Images / Sec)", fontsize=11)
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels([str(b) for b in batch_sizes])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Panel 2: Batch-1 Latency (Lower is Better)
    frameworks = ["PyTorch (MPS)", "Apple MLX", "Candle (Rust)"]
    p50_vals = [torch_lat["p50"], mlx_lat["p50"], candle_lat["p50"]]
    p95_vals = [torch_lat["p95"], mlx_lat["p95"], candle_lat["p95"]]
    bar_x = np.arange(len(frameworks))
    width = 0.35

    b1 = axes[1].bar(bar_x - width/2, p50_vals, width, label="P50 (Median Latency)", color="#34C759")
    b2 = axes[1].bar(bar_x + width/2, p95_vals, width, label="P95 (Tail Latency)", color="#FF9500")
    axes[1].set_title("Single-Image Inference Latency (Batch Size = 1)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Latency (Microseconds - Lower is Better)", fontsize=11)
    axes[1].set_xticks(bar_x)
    axes[1].set_xticklabels(frameworks, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=10)

    for bar in b1:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval:.0f}μs", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval:.0f}μs", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    print(f"\n[✓] 3-Way Shootout Plot saved to: {os.path.abspath(output_plot)}")

if __name__ == "__main__":
    main()
