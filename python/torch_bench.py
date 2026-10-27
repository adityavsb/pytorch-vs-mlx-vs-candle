"""
PyTorch Benchmark Module on Apple Silicon MPS (Metal Performance Shaders).
Measures:
1. Inference (forward-only, no-grad) throughput (images/sec) across batch sizes
2. Microsecond Inference Latency at Batch Size 1 (P50, P95, P99)
3. Python + Metal Dispatch Floor Latency
"""
import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def benchmark_torch_dispatch_floor(iterations=500):
    device = get_device()
    identity = nn.Identity().to(device)
    x = torch.randn(1, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(20):
            _ = identity(x)
        if device.type == "mps":
            torch.mps.synchronize()

        latencies_us = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _ = identity(x)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)

    p50 = float(np.percentile(latencies_us, 50))
    return p50

def benchmark_torch_throughput(batch_sizes=[32, 64, 128, 256, 512], min_seconds=2.0, min_iters=50, compile=False, verbose=True):
    device = get_device()
    model = StandardCNN().to(device)
    model.eval()

    if compile:
        model = torch.compile(model)

    results = {}
    engine_label = "PyTorch (MPS, torch.compile)" if compile else f"PyTorch on {device}"
    if verbose:
        print(f"[{engine_label}] Benchmarking Throughput...")

    warmup_iters = 10 if compile else 5
    step_chunk = 20
    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.randn(bs, 1, 28, 28, device=device)
            # Warmup
            for _ in range(warmup_iters):
                _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()

            n = 0
            start_t = time.perf_counter()
            while True:
                for _ in range(step_chunk):
                    _ = model(x)
                if device.type == "mps":
                    torch.mps.synchronize()
                n += step_chunk
                if n >= min_iters and (time.perf_counter() - start_t) >= min_seconds:
                    break
            duration = time.perf_counter() - start_t

            total_images = bs * n
            img_per_sec = total_images / duration
            results[bs] = img_per_sec
            if verbose:
                print(f"  Batch {bs:3d}: {img_per_sec:8.0f} img/s ({duration/n*1000:6.2f} ms/step, {n} iters in {duration:.2f}s)")

    return results

def benchmark_torch_latency_batch1(iterations=500, compile=False, verbose=True):
    device = get_device()
    model = StandardCNN().to(device)
    model.eval()

    if compile:
        model = torch.compile(model)

    x = torch.randn(1, 1, 28, 28, device=device)
    warmup_iters = 30 if compile else 20
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()

        latencies_us = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0) # microseconds

    p50 = float(np.percentile(latencies_us, 50))
    p95 = float(np.percentile(latencies_us, 95))
    p99 = float(np.percentile(latencies_us, 99))
    if verbose:
        engine_label = "PyTorch (MPS, torch.compile)" if compile else "PyTorch"
        print(f"[{engine_label} Batch-1 Latency] P50: {p50:6.1f} μs | P95: {p95:6.1f} μs | P99: {p99:6.1f} μs")
    return {"p50": p50, "p95": p95, "p99": p99}

def main():
    parser = argparse.ArgumentParser(description="PyTorch Apple Silicon Benchmark Module")
    parser.add_argument("--json", action="store_true", help="Output JSON_RESULT line for orchestrator")
    parser.add_argument("--compile", action="store_true", help="Benchmark with torch.compile enabled")
    parser.add_argument("--floor", action="store_true", help="Measure Python dispatch floor latency")
    parser.add_argument("--min-seconds", type=float, default=2.0, help="Minimum measurement window per throughput cell")
    parser.add_argument("--min-iters", type=int, default=50, help="Minimum iterations per throughput cell")
    parser.add_argument("--latency-iters", type=int, default=500, help="Iterations for batch-1 latency benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512], help="Batch sizes to benchmark")
    args = parser.parse_args()

    verbose = not args.json
    tput = benchmark_torch_throughput(
        batch_sizes=args.batch_sizes,
        min_seconds=args.min_seconds,
        min_iters=args.min_iters,
        compile=args.compile,
        verbose=verbose,
    )
    lat = benchmark_torch_latency_batch1(
        iterations=args.latency_iters,
        compile=args.compile,
        verbose=verbose,
    )

    floor_us = benchmark_torch_dispatch_floor() if args.floor else None
    if verbose and floor_us is not None:
        print(f"[PyTorch Dispatch Floor] Identity P50: {floor_us:6.1f} μs")

    if args.json:
        engine_name = "PyTorch (MPS, torch.compile)" if args.compile else "PyTorch (Apple MPS)"
        payload = {
            "engine": engine_name,
            "throughput": {str(k): float(v) for k, v in tput.items()},
            "latency_us": lat,
        }
        if floor_us is not None:
            payload["dispatch_floor_us"] = floor_us
        print("JSON_RESULT: " + json.dumps(payload))

if __name__ == "__main__":
    main()
