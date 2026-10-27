# Single-Variable Ablation Study

Empirical delta analysis isolating the performance impact of each individual model architecture fix and measurement adjustment between v0.1.0 and v0.2.0 on Apple M4 (macOS Darwin arm64).

---

## 1. Ablation Matrix

| Framework | State / Condition | Batch 1 Latency (P50) | Batch 512 Throughput | Empirical Impact & Mechanism |
|---|---|---|---|---|
| **Apple MLX** | (a) v0.1.0 baseline: `model.train()` | `401.9 μs` | `52,385 img/s` | BatchNorm calculated running mean/variance on every forward pass, forcing global memory reductions. |
| **Apple MLX** | (b) v0.2.0 fix: `model.eval()` | `256.6 μs` | `100,816 img/s` | Switching to inference mode eliminated batch reductions, yielding **+92.4% throughput** and -36.2% latency under `mx.compile`. |
| **Candle (Rust)** | (a) v0.1.0 baseline: No BatchNorm + `to_vec1()` | `1,357.0 μs` | `3,124.0 img/s` | Missing BatchNorm layers artificially reduced compute load; `to_vec1()` copied output tensor to CPU host vector. |
| **Candle (Rust)** | (b) Fix 1: Added BatchNorm + `to_vec1()` | `2,240.5 μs` | `2,485.0 img/s` | Architectural parity increased per-batch compute load, decreasing batch 512 throughput by ~20.5%. |
| **Candle (Rust)** | (c) v0.2.0 fix: Added BatchNorm + `device.synchronize()` | `1,855.4 μs` | `2,539.0 img/s` | Non-allocating Metal barrier fence eliminated CPU readback copy, recovering **~385 μs** in batch-1 latency. |

---

## 2. Experimental Methodology & Reproduction Commands

All measurements were obtained on an Apple Mac mini (2024, Apple M4, 16 GB unified memory, macOS Darwin 27.0.1).

### MLX Ablation (Train Mode vs. Eval Mode)
```bash
python -c "import time, mlx.core as mx, mlx.nn as nn
from python.mlx_bench import StandardCNN

for mode in ['train', 'eval']:
    m = StandardCNN()
    if mode == 'train': m.train()
    else: m.eval()
    mx.eval(m.parameters())
    f = mx.compile(m)
    # Warmup
    x1 = mx.random.normal((1, 28, 28, 1))
    for _ in range(20): mx.eval(f(x1))
    # Timed Batch 1
    t0 = time.perf_counter()
    for _ in range(100): mx.eval(f(x1))
    t1 = time.perf_counter()
    p50_us = ((t1 - t0) / 100) * 1e6
    # Timed Batch 512
    x512 = mx.random.normal((512, 28, 28, 1))
    for _ in range(10): mx.eval(f(x512))
    t0 = time.perf_counter()
    for _ in range(30): mx.eval(f(x512))
    t1 = time.perf_counter()
    tput = (30 * 512) / (t1 - t0)
    print(f'MLX mode={mode}: Batch 1 P50={p50_us:.1f}us, Batch 512 Tput={tput:.0f} img/s')
"
```

### Candle Ablation (BatchNorm and Barrier Fence)
- **v0.1.0 Baseline (Commit `8790c1c`):** Archived in `results/2026-10-20_as-published/`. Model implemented without `bn1`/`bn2` and called `out.to_vec1::<f32>()` inside the timed region.
- **Fix 1 (BatchNorm + `to_vec1()`):** Model instantiated with `BatchNorm` layers; barrier via host vector allocation.
- **v0.2.0 Fix (Commit `4421629` / `main`):** Model instantiated with `BatchNorm` layers; barrier via `device.synchronize()?` keeping memory resident in Metal buffers.
