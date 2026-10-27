"""
Apple MLX Benchmark Module on Apple Silicon Unified Memory.
Measures:
1. Inference (forward-only, no-grad) throughput (images/sec) across batch sizes
2. Microsecond Inference Latency at Batch Size 1 (P50, P95, P99)
3. Python + MLX Dispatch Floor Latency
"""
import argparse
import json
import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

class StandardCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x):
        # x is (N, H, W, C)
        x = self.pool(nn.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.relu(self.bn2(self.conv2(x))))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.fc1(x))
        return self.fc2(x)

def benchmark_mlx_dispatch_floor(iterations=500):
    x = mx.random.normal((1, 28, 28, 1))
    identity = lambda val: val
    # Warmup
    for _ in range(20):
        out = identity(x)
        mx.eval(out)

    latencies_us = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        out = identity(x)
        mx.eval(out)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    p50 = float(np.percentile(latencies_us, 50))
    return p50

def benchmark_mlx_throughput(batch_sizes=[32, 64, 128, 256, 512], min_seconds=2.0, min_iters=50, verbose=True):
    model = StandardCNN()
    model.eval()  # inference-mode BatchNorm: use running stats, no stat updates
    mx.eval(model.parameters())
    
    compiled_model = mx.compile(model)
    results = {}
    if verbose:
        print(f"[Apple MLX] Benchmarking Throughput on device: {mx.default_device()}")

    for bs in batch_sizes:
        # MLX expects (N, H, W, C)
        x = mx.random.normal((bs, 28, 28, 1))
        # Warmup and compile graph
        for _ in range(5):
            out = compiled_model(x)
            mx.eval(out)

        n = 0
        start_t = time.perf_counter()
        while True:
            out = compiled_model(x)
            mx.eval(out)
            n += 1
            if n >= min_iters and (time.perf_counter() - start_t) >= min_seconds:
                break
        duration = time.perf_counter() - start_t

        total_images = bs * n
        img_per_sec = total_images / duration
        results[bs] = img_per_sec
        if verbose:
            print(f"  Batch {bs:3d}: {img_per_sec:8.0f} img/s ({duration/n*1000:6.2f} ms/step, {n} iters in {duration:.2f}s)")

    return results

def benchmark_mlx_latency_batch1(iterations=500, verbose=True):
    model = StandardCNN()
    model.eval()
    mx.eval(model.parameters())
    compiled_model = mx.compile(model)

    x = mx.random.normal((1, 28, 28, 1))
    # Warmup
    for _ in range(20):
        out = compiled_model(x)
        mx.eval(out)

    latencies_us = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        out = compiled_model(x)
        mx.eval(out)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0) # microseconds

    p50 = float(np.percentile(latencies_us, 50))
    p95 = float(np.percentile(latencies_us, 95))
    p99 = float(np.percentile(latencies_us, 99))
    if verbose:
        print(f"[Apple MLX Batch-1 Latency] P50: {p50:6.1f} μs | P95: {p95:6.1f} μs | P99: {p99:6.1f} μs")
    return {"p50": p50, "p95": p95, "p99": p99}

def main():
    parser = argparse.ArgumentParser(description="Apple MLX Silicon Benchmark Module")
    parser.add_argument("--json", action="store_true", help="Output JSON_RESULT line for orchestrator")
    parser.add_argument("--floor", action="store_true", help="Measure Python dispatch floor latency")
    parser.add_argument("--min-seconds", type=float, default=2.0, help="Minimum measurement window per throughput cell")
    parser.add_argument("--min-iters", type=int, default=50, help="Minimum iterations per throughput cell")
    parser.add_argument("--latency-iters", type=int, default=500, help="Iterations for batch-1 latency benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512], help="Batch sizes to benchmark")
    args = parser.parse_args()

    verbose = not args.json
    tput = benchmark_mlx_throughput(
        batch_sizes=args.batch_sizes,
        min_seconds=args.min_seconds,
        min_iters=args.min_iters,
        verbose=verbose,
    )
    lat = benchmark_mlx_latency_batch1(
        iterations=args.latency_iters,
        verbose=verbose,
    )

    floor_us = benchmark_mlx_dispatch_floor() if args.floor else None
    if verbose and floor_us is not None:
        print(f"[Apple MLX Dispatch Floor] Identity P50: {floor_us:6.1f} μs")

    if args.json:
        payload = {
            "engine": "Apple MLX (Native GPU)",
            "throughput": {str(k): float(v) for k, v in tput.items()},
            "latency_us": lat,
        }
        if floor_us is not None:
            payload["dispatch_floor_us"] = floor_us
        print("JSON_RESULT: " + json.dumps(payload))

if __name__ == "__main__":
    main()
