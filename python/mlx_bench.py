"""
Apple MLX Benchmark Module on Apple Silicon Unified Memory.
Measures:
1. Training and Forward Throughput (images/sec) across batch sizes
2. Microsecond Inference Latency at Batch Size 1 (P50, P95, P99)
3. Active GPU Unified Memory Allocation (MB)
"""
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

def benchmark_mlx_throughput(batch_sizes=[32, 64, 128, 256, 512], iterations=40):
    model = StandardCNN()
    mx.eval(model.parameters())
    
    compiled_model = mx.compile(model)
    results = {}
    print(f"[Apple MLX] Benchmarking Throughput on device: {mx.default_device()}")

    for bs in batch_sizes:
        # MLX expects (N, H, W, C)
        x = mx.random.normal((bs, 28, 28, 1))
        # Warmup and compile graph
        for _ in range(5):
            out = compiled_model(x)
            mx.eval(out)

        start_t = time.perf_counter()
        for _ in range(iterations):
            out = compiled_model(x)
            mx.eval(out)
        duration = time.perf_counter() - start_t

        total_images = bs * iterations
        img_per_sec = total_images / duration
        results[bs] = img_per_sec
        print(f"  Batch {bs:3d}: {img_per_sec:8.0f} img/s ({duration/iterations*1000:6.2f} ms/step)")

    mem_mb = 0.0
    if hasattr(mx, "get_active_memory"):
        mem_mb = mx.get_active_memory() / (1024 * 1024)
    elif hasattr(mx.metal, "get_active_memory"):
        mem_mb = mx.metal.get_active_memory() / (1024 * 1024)

    return results, mem_mb

def benchmark_mlx_latency_batch1(iterations=500):
    model = StandardCNN()
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
    print(f"[Apple MLX Batch-1 Latency] P50: {p50:6.1f} μs | P95: {p95:6.1f} μs | P99: {p99:6.1f} μs")
    return {"p50": p50, "p95": p95, "p99": p99}

if __name__ == "__main__":
    benchmark_mlx_throughput()
    benchmark_mlx_latency_batch1()
