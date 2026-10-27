# Changelog

All notable changes to the `pytorch-vs-mlx-vs-candle` benchmark repository are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-10-27

### Author Reflection & Methodology Revision
*By Aditya Bhadoria*

When I initially published version `0.1.0` of this benchmark, my goal was to quickly explore how PyTorch MPS, Apple MLX, and Hugging Face Candle in Rust compared out of the box on my new Apple M4 Mac mini. However, a deeper audit revealed critical methodology and architectural parity issues: Candle was missing BatchNorm layers while PyTorch and MLX included them; MLX was running in default training mode which forced unnecessary batch statistics reductions; Candle's timing loop included a host CPU copy (`to_vec1`) inside the timed region; and results were based on a single unrepeated run.

In this `0.2.0` release, I resolved every disparity to ensure true architectural and measurement parity. All three models now implement identical LeNet-scale CNN architectures with inference-mode BatchNorm, timing loops fence Metal commands without host copies, and numbers reflect medians and min–max spreads across 5 randomized trials. While compiled MLX demonstrates significant latency advantages on this dispatch-bound workload, I want to emphasize that these measurements reflect a toy model (~8.5 MFLOP/image) in its default configuration and cannot be generalized to large, compute-bound neural networks such as ResNet-50 or transformers.

---

### Before & After: Version Comparison (v0.1.0 vs v0.2.0)

| Metric | Framework | v0.1.0 (As Published, Single Run) | v0.2.0 (Verified, 5-Trial Median) | Delta / Explanation |
|---|---|---|---|---|
| **Batch 512 Throughput** | PyTorch (MPS) | `61,048 img/s` | `63,159 img/s` | +3.5% (Duration-based timing >=2.0s) |
| **Batch 512 Throughput** | Apple MLX (Native) | `52,190 img/s` | `101,415 img/s` | **+94.3%** (`model.eval()` eliminated train reductions) |
| **Batch 512 Throughput** | Candle (Rust) | `3,124 img/s` | `2,539 img/s` | -18.7% (Added BatchNorm layers to model) |
| **Batch 1 Latency P50** | PyTorch (MPS) | `462.1 μs` | `458.0 μs` | Consistent eager command dispatch |
| **Batch 1 Latency P50** | Apple MLX (Native) | `321.4 μs` | `275.8 μs` | -14.2% (`model.eval()` static graph fusion) |
| **Batch 1 Latency P50** | Candle (Rust) | `1,357.0 μs` | `1,855.4 μs` | BatchNorm added; `device.synchronize()` barrier |

---

### Single-Variable Ablation Matrix

To isolate the exact performance impact of each individual bug fix, I measured each state independently on the Apple M4:

| Framework | State / Condition | Batch 1 Latency (P50) | Batch 512 Throughput | Empirical Impact & Mechanism |
|---|---|---|---|---|
| **Apple MLX** | (a) v0.1.0 baseline: `model.train()` | `401.9 μs` | `52,385 img/s` | BatchNorm actively calculated running statistics on every forward pass, forcing global memory reductions. |
| **Apple MLX** | (b) v0.2.0 fix: `model.eval()` | `256.6 μs` | `100,816 img/s` | Switching to inference mode eliminated batch reductions, yielding **+92.4% throughput** under `mx.compile`. |
| **Candle (Rust)** | (a) v0.1.0 baseline: No BatchNorm + `to_vec1()` | `1,357.0 μs` | `3,124.0 img/s` | Missing BatchNorm layers artificially reduced compute load; `to_vec1()` copied output tensor to CPU host vector. |
| **Candle (Rust)** | (b) Fix 1: Added BatchNorm + `to_vec1()` | `2,240.5 μs` | `2,485.0 img/s` | Architectural parity increased per-batch compute load, decreasing batch 512 throughput by ~20.5%. |
| **Candle (Rust)** | (c) v0.2.0 fix: Added BatchNorm + `device.synchronize()` | `1,855.4 μs` | `2,539.0 img/s` | Non-allocating Metal barrier fence eliminated CPU readback copy, recovering **~385 μs** in batch-1 latency. |

---

### Detailed Changes in v0.2.0

#### Benchmark Logic & Parity
- **Candle BatchNorm Parity:** Added `bn1` and `bn2` (`candle_nn::batch_norm`) in eval mode (`forward_t(..., false)`), matching PyTorch and MLX architecture.
- **MLX Evaluation Mode:** Explicitly invoked `model.eval()` before benchmarking to prevent training-mode running mean and variance updates.
- **Non-Allocating Metal Barrier:** Replaced Candle CPU-roundtrip readback (`to_vec1::<f32>()`) with `device.synchronize()?` barrier fence.
- **Timing Loops:** Replaced fixed 40 iterations with duration-paced loops (`>= 2.0s`, `>= 50 iterations`) with `step_chunk = 20` to prevent GPU command buffer queue backlog.
- **Percentiles:** Standardized latency percentile calculation to linear interpolation matching NumPy.
- **Dropped Memory Row:** Removed non-comparable GPU memory footprint metrics.
- **Baseline Snapshot:** Archived immutable baseline in `results/2026-10-20_as-published/`.

#### Methodology & Statistical Rigor
- **Multi-Trial Harness:** Implemented 5-trial randomized execution harness with rotated framework order.
- **Subprocess Isolation:** Isolated PyTorch, MLX, and Candle executions in separate subprocesses via standardized `JSON_RESULT: {...}` stdout interface.
- **Machine Telemetry:** Logged macOS thermal status (`pmset -g therm`) before and after sweep.
- **Data Export:** Exported complete multi-trial raw runs and summary statistics to `results/latest.json`.
- **Environment Pinning:** Pinned `requirements.txt` dependencies, tracked `rust_candle/Cargo.lock`, and pinned `rust-toolchain.toml` to 1.99.1.
- **Dispatch Floor Baseline:** Measured Python identity dispatch floor (~0.58 μs for PyTorch MPS, ~0.17 μs for MLX).

#### Knowledge, Literature & Reputation
- **Model Code Parity:** Aligned README code blocks with actual `StandardCNN` models including BatchNorm and pooling.
- **Mechanisms Grounding:** Replaced marketing superlatives with documented compiler and kernel lowering mechanisms.
- **Citations:** Added 6 formal references (Hannun et al., Paszke et al., MPSGraph, Candle, M4 Mac mini specs).
- **Scope & Limitations:** Added comprehensive 6-point disclosure covering workload bounds, execution modes, repetition, unmeasured dimensions, machine state, and layer layouts.
- **Citation Metadata:** Added standard `CITATION.cff` (v0.2.0).
- **Smoke Tests & CI:** Added shape smoke tests (`tests/test_shapes.py` and Rust test in `main.rs`) and GitHub Actions workflow (`.github/workflows/ci.yml`).
- **README Validator:** Created `scripts/check_readme.py` to assert that documentation numbers remain synchronized with benchmark output.

---

## [0.1.0] — 2026-10-20

### Added
- Initial 3-way benchmark comparing PyTorch (MPS), Apple MLX, and Hugging Face Candle on Apple Silicon M4.
- Basic throughput and batch-1 latency comparison plots.
