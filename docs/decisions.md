# Architecture Decision Records (ADRs)

Engineering decisions, architectural trade-offs, and design rationale for the `pytorch-vs-mlx-vs-candle` Apple Silicon benchmark.

---

## ADR-001: Benchmark Architecture — LeNet-Scale CNN vs. Compute-Bound Models

### Status
Accepted

### Context
When benchmarking deep learning runtimes on unified memory architectures (Apple Silicon), test workloads typically fall into one of two regimes:
1. **Compute-Bound:** Large matrix multiplications (GEMM) such as LLaMA transformer blocks or ResNet-50 on ImageNet, where arithmetic intensity ($FLOPS / Byte$) is high and performance is determined by hardware tensor cores or SIMD compute density.
2. **Dispatch-Bound (Overhead-Bound):** Small to medium layers where execution time per op is on the order of tens to hundreds of microseconds, and performance is determined by runtime host dispatch, Metal command encoding, kernel fusion, and driver queue latency.

### Decision
We benchmark a two-layer convolutional network (LeNet-scale: $2\times$ `Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool` $\rightarrow$ `Dense`) on single-channel $28\times 28$ inputs (~8.5 MFLOP per image).

### Rationale
- High-compute models measure Apple M4 raw hardware TFLOP/s, which is already well-documented by Apple and standard SPEC benchmarks.
- In practical deployments (such as interactive robotics, real-time audio, and microsecond perception loops), model latency is dominated by host overhead and synchronization barriers.
- A LeNet-scale network directly isolates the software runtime differences: MLX's graph compiler vs. PyTorch's eager Metal Performance Shaders Graph lowering vs. Candle's pure systems-level Rust Metal dispatch.

### Consequences & Trade-offs
- **Limitation:** The results do not generalize to transformer inference or large vision backbones. We explicitly document this in `README.md` under [Scope and Limitations](../README.md#scope-and-limitations).

---

## ADR-002: Duration-Based Benchmarking with Chunked GPU Synchronization

### Status
Accepted (Supersedes fixed 40-iteration timing loops)

### Context
In v0.1.0, the benchmark measured 40 iterations. At microsecond latencies, 40 iterations completed in under 30 ms. This introduced two severe measurement errors:
1. Timing inside transient CPU boost clocks before thermal equilibrium.
2. High clock jitter relative to total run duration.

However, naive duration loops (`while elapsed < 2.0s: model(x)`) introduced an even worse failure mode on Apple Metal: because Python CPU execution dispatches asynchronously, the CPU submitted >8,000 passes to the Metal command buffer queue in 2 seconds. When the barrier fence `torch.mps.synchronize()` was finally called outside the loop, the GPU took over 90 seconds to drain the backlogged queue, producing distorted throughput calculations.

### Decision
We implement duration-based timing loops (`>= 2.0s` and `>= 50 iterations`) with **chunked command buffer synchronization** (`step_chunk = 20`):
```python
while elapsed < min_duration or total_iters < min_iters:
    for _ in range(step_chunk):
        out = model(x)
    torch.mps.synchronize()
    total_iters += step_chunk
    elapsed = time.perf_counter() - start_time
```

### Rationale
- Chunks of 20 iterations allow the GPU pipeline to remain saturated without CPU stall bubbles.
- Periodic synchronization prevents the host driver command buffer from overflowing.
- Benchmarks sustain consistent steady-state execution for at least 2.0 seconds, yielding statistically repeatable throughput numbers.

---

## ADR-003: Multi-Trial Execution with Randomized Framework Order

### Status
Accepted

### Context
Apple Silicon unified memory chips feature aggressive dynamic power and thermal management. Running framework sweeps sequentially (PyTorch all batch sizes $\rightarrow$ MLX all batch sizes $\rightarrow$ Candle all batch sizes) biases results:
- The first framework executes on cold silicon with maximum clock headroom.
- The final framework executes on warm silicon with potential thermal throttling.
- Background OS processes (WindowServer, Metal driver memory caching) can skew single-run measurements.

### Decision
We run 5 independent trials (`--trials 5`), rotating the execution order on each trial (`frameworks[trial % len(frameworks)]`), with a 10-second cooldown between framework runs, and report the **median and min–max spread**:
$$\text{Reported Metric} = \text{Median}(T_1, \dots, T_5) \quad [\text{Min} - \text{Max}]$$

### Rationale
- Eliminates ordering bias and dampens transient OS spikes.
- Min–max spread gives immediate visual confidence in benchmark stability.
- Results are saved to a machine-readable JSON schema (`results/latest.json`) with before/after thermal state logs (`pmset -g therm`).

---

## ADR-004: In-GPU Device Barrier Fence vs. Host Memory Readback

### Status
Accepted (Supersedes `to_vec1::<f32>()` CPU readback in Candle)

### Context
To benchmark pure GPU inference latency, all asynchronous command buffers submitted to Metal must complete before taking the wall-clock timestamp.
In v0.1.0, Candle used:
```rust
let out = model.forward(&x)?;
let _ = out.to_vec1::<f32>()?; // Readback to CPU
```
This conflated GPU kernel compute time with CPU memory allocation (`Vec<f32>::with_capacity`) and bus transfer overhead.

### Decision
We synchronize directly on the Metal command buffer via `device.synchronize()?` across all frameworks:
- **PyTorch:** `torch.mps.synchronize()`
- **MLX:** `mx.eval(out)`
- **Candle:** `device.synchronize()?`

### Rationale
- Keeps tensors resident in Apple Silicon Unified Memory without unnecessary CPU memory copies.
- Ensures all frameworks measure the exact same physical milestone: GPU kernel execution complete and command buffer retired.
