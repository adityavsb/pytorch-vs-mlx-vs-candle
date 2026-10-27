"""
PyTorch Benchmark Module on Apple Silicon MPS (Metal Performance Shaders).
Measures:
1. Inference (forward-only, no-grad) throughput (images/sec) across batch sizes
2. Microsecond Inference Latency at Batch Size 1 (P50, P95, P99)
"""
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

def benchmark_torch_throughput(batch_sizes=[32, 64, 128, 256, 512], min_seconds=2.0, min_iters=50):
    device = get_device()
    model = StandardCNN().to(device)
    model.eval()

    results = {}
    print(f"[PyTorch] Benchmarking Throughput on device: {device}")

    step_chunk = 20
    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.randn(bs, 1, 28, 28, device=device)
            # Warmup
            for _ in range(5):
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
            print(f"  Batch {bs:3d}: {img_per_sec:8.0f} img/s ({duration/n*1000:6.2f} ms/step, {n} iters in {duration:.2f}s)")

    return results

def benchmark_torch_latency_batch1(iterations=500):
    device = get_device()
    model = StandardCNN().to(device)
    model.eval()

    x = torch.randn(1, 1, 28, 28, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(20):
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
    print(f"[PyTorch Batch-1 Latency] P50: {p50:6.1f} μs | P95: {p95:6.1f} μs | P99: {p99:6.1f} μs")
    return {"p50": p50, "p95": p95, "p99": p99}

if __name__ == "__main__":
    benchmark_torch_throughput()
    benchmark_torch_latency_batch1()
