# Apple Silicon Deep Learning Shootout: PyTorch vs. Apple MLX vs. Hugging Face Candle (Rust)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.99.1-orange.svg)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(MPS)-EE4C2C.svg)](https://pytorch.org/)
[![Apple MLX](https://img.shields.io/badge/Apple%20MLX-0.32%20(Native%20GPU)-0071E3.svg)](https://github.com/ml-explore/mlx)
[![Candle](https://img.shields.io/badge/HF%20Candle-0.11%20(Rust%20Metal)-yellow.svg)](https://github.com/huggingface/candle)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20M4%20Mac%20mini-black.svg)](https://support.apple.com/en-us/121555)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A reproducible micro-benchmark comparing three deep learning frameworks head-to-head on the **Apple M4 Mac mini**:

1. **PyTorch (Python / MPS):** PyTorch using Apple's Metal Performance Shaders (MPS) backend.
2. **Apple MLX (Python / C++ Metal):** Apple ML Research's framework engineered from scratch for unified memory zero-copy operations.
3. **Hugging Face Candle (Rust / Metal):** A minimalist systems-level framework compiled in pure Rust, eliminating Python's GIL and runtime overhead.

> **Note on Methodology & History:** See [CHANGELOG.md](CHANGELOG.md) for the v0.1.0 → v0.2.0 revision details, including an empirical single-variable ablation of each fix.

---

## What I learned

Running this three-framework shootout on the M4 Mac mini challenged several of my initial assumptions about systems-level machine learning:

1. **Compiled Rust does not automatically beat Python on GPU:** I expected Candle in pure compiled Rust to lead across the board by eliminating Python runtime overhead. In reality, on dispatch-bound GPU workloads, framework performance is governed by how effectively operations are lowered to Metal shaders. Candle's Metal backend lowers `conv2d` via an un-fused im2col buffer allocation followed by GEMM, which causes throughput to scale downwards as batch size increases (`3,069 img/s` at batch 32 down to `2,539 img/s` at batch 512 in `results/latest.json`). Apple MLX, by contrast, traces the graph under `mx.compile` and executes a single fused Metal pipeline, reaching `101,415 img/s`.
2. **The subtleties of asynchronous GPU command buffers:** When transitioning to duration-based loops (`>= 2.0s`), unpaced Python loops queued over 8,000 passes into the Metal driver in two seconds, freezing the GPU for 90 seconds when the barrier fence was finally called. Chunking submissions with `step_chunk = 20` was necessary to pipeline GPU command buffers while keeping driver dispatch paced to wall-clock time.
3. **The impact of inference vs. training mode:** In early iterations, omitting `model.eval()` in MLX silently triggered dynamic batch statistics reductions on every forward pass, cutting throughput in half (`52,385 img/s` vs `101,415 img/s`). Verifying layer invariants and parameter counts via automated tests is critical before drawing performance conclusions.

If I were continuing this investigation, I would capture Metal System Trace timeline traces in Apple Instruments to inspect hardware threadgroup occupancy and shader stall cycles directly on the M4 GPU cores, and extend the comparison to compute-bound architectures where FLOP density dominates driver overhead.

---

## Hardware & Testbench Specifications

```text
Machine:             Apple Mac mini (2024)
SoC:                 Apple M4
GPU:                 Apple M4 (10-core GPU, Metal 4)
Unified Memory:      16.00 GB (Shared CPU/GPU)
Platform / OS:       macOS Darwin (arm64)
Rust Toolchain:      rustc 1.99.1 / cargo 1.99.1
Python Runtime:      Python 3.13.6
PyTorch Backend:     torch 2.14.1 (MPS - Metal Performance Shaders)
Apple MLX Backend:   mlx 0.32.4 (Device: gpu, 0)
HF Candle Backend:   candle-core 0.11.0 (Metal)
```

---

## Benchmark Results

All three frameworks were tested on an identical two-layer convolutional network (LeNet-scale: $2\times$ `Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool` $\rightarrow$ `Dense`):

> **Execution modes:** PyTorch runs eager (`torch.compile` optional); MLX runs under `mx.compile`; Candle dispatches eagerly. The comparison is therefore *default-configuration* per framework, not best-achievable per framework.

### 1. Throughput & Latency Summary (Medians over 5 Trials)

| Metric | PyTorch (MPS) | Apple MLX (Native) | Candle (Rust Metal) | Comparison Note |
| :--- | :---: | :---: | :---: | :--- |
| **Batch 32 Throughput** | `45,783 img/s` | `52,969 img/s` | `3,069 img/s` | MLX +16% vs PyTorch |
| **Batch 64 Throughput** | `55,091 img/s` | `65,087 img/s` | `2,978 img/s` | MLX +18% vs PyTorch |
| **Batch 128 Throughput** | `59,621 img/s` | `84,325 img/s` | `2,797 img/s` | MLX +41% vs PyTorch |
| **Batch 256 Throughput** | `60,930 img/s` | `93,798 img/s` | `2,640 img/s` | MLX +54% vs PyTorch |
| **Batch 512 Throughput** | `63,159 img/s` | `101,415 img/s` | `2,539 img/s` | MLX +61% vs PyTorch |
| **Batch-1 Latency (P50 Median)** | `458.0 μs` | `275.8 μs` | `1855.4 μs` | MLX 1.66× faster vs PyTorch |
| **Batch-1 Latency (P95 Tail)** | `471.1 μs` | `285.6 μs` | `2280.5 μs` | MLX 1.65× faster vs PyTorch |
| **Batch-1 Latency (P99 Max)** | `569.1 μs` | `332.6 μs` | `2659.5 μs` | MLX 1.71× faster vs PyTorch |

![Shootout Results](assets/shootout_results.png)

---

## Key Systems Engineering Takeaways

### 1. Batch-1 Latency: Compiled MLX Leads Eager PyTorch
* **Why MLX leads at batch 1 (in this configuration):** The MLX model is run under `mx.compile`, which traces the forward pass once and dispatches a fused kernel graph on each call; PyTorch is run eager, issuing one Metal command encoder per op. The gap therefore measures *compiled MLX vs eager PyTorch*, not MLX vs PyTorch in general. See [MLX Compilation Documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

### 2. Throughput Scaling: Compiled MLX Leads Across Tested Batch Sizes
* **Why PyTorch throughput scales:** PyTorch's MPS backend lowers operations to Apple's Metal Performance Shaders Graph framework ([MPSGraph Documentation](https://pytorch.org/docs/stable/notes/mps.html)). Kernel-level behaviour was not profiled with Metal System Trace, and no claim is made about internal MPSGraph shader scheduling.

### 3. Hugging Face Candle (Rust Metal): Trade-offs in Unfused Systems Deployment
* **Why Candle trails on throughput:** Candle's Metal `conv2d` is implemented as an im2col transformation followed by a GEMM kernel (`candle-core/src/metal_backend/mod.rs`), dispatched op-by-op without graph fusion; throughput falls as batch size grows, consistent with an im2col buffer that scales with batch size. Candle's primary advantages — a single static binary and no Python runtime — were not timed in this loop.

---

## Scope and limitations

- **Workload.** A two-conv, two-FC network on 28×28 single-channel random inputs (~8.5 MFLOP per image). At the measured throughput the GPU sustains roughly 0.5 TFLOP/s, far below M4 peak: this benchmark is **dispatch-bound**, and results do not transfer to compute-bound models such as ResNet-50 or transformer inference.
- **Execution modes.** PyTorch eager (and, where shown, `torch.compile`); MLX under `mx.compile`; Candle eager. Framework and compilation mode are confounded unless both PyTorch columns are present.
- **Repetition.** Every number is the median of 5 trials with min–max in parentheses; the raw per-trial data is in `results/latest.json`. Earlier versions of this README reported a single run.
- **What was not measured.** Training, backward pass, memory footprint, model loading, multi-model concurrency, larger input sizes, power draw, or any real dataset.
- **Machine state.** One Mac mini (2024, M4, 16 GB), macOS version and framework versions recorded in `results/latest.json`; no thermal control beyond a 10 s idle between frameworks.
- **Models.** All three models have the same layer sequence (Conv-BN-ReLU-Pool ×2, FC-ReLU-FC) and run BatchNorm in inference mode. The MLX model uses NHWC layout as required by `mlx.nn.Conv2d`; PyTorch and Candle use NCHW.

## Engineering Friction & Where Theory Met Silicon

Real hardware systems reveal architectural friction that theoretical benchmarks obscure:

1. **The Metal Command Buffer Queue Backlog:** In duration-based benchmarking (`>= 2.0s`), unpaced Python CPU loops queued >8,000 passes into the Metal driver queue within 2 seconds. When the barrier fence `torch.mps.synchronize()` was invoked outside the loop, the GPU required over 90 seconds to drain the backlogged queue, distorting throughput calculations. Introducing chunked synchronization (`step_chunk = 20`) pipelined execution while pacing driver dispatch to wall-clock time.
2. **Why Compiled Rust Trailed Compiled Python on Metal:** Pure Rust eliminates Python runtime overhead, leading us to expect Candle to lead or match MLX. In reality, Candle's Metal backend lowers `conv2d` via an un-fused im2col buffer allocation followed by GEMM (`candle-core/src/metal_backend/mod.rs`). As batch size grows to 512, this un-fused buffer allocation saturates memory bandwidth, causing throughput to fall from 3,069 img/s down to 2,539 img/s. MLX, conversely, traces operations once under `mx.compile` and executes a single fused Metal compute pipeline.
3. **The 92% Training-Mode Cliff:** In our initial v0.1.0 run, omitting `model.eval()` caused MLX to calculate per-batch running mean and variance reductions on every inference step, cutting throughput by nearly half (`52,385 img/s` vs `101,415 img/s` in eval mode).

> Detailed architectural decisions and rejected alternatives are recorded in [docs/decisions.md](docs/decisions.md).

## Results file

Complete multi-trial run outputs, per-framework distributions, and system telemetry are exported to [`results/latest.json`](results/latest.json).

---

## Code Comparison Across Frameworks

These are the exact models timed; see the source files for the timing loops.

### 1. PyTorch (Python / MPS)
```python
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = StandardCNN().to(device)
model.eval()
x = torch.randn(128, 1, 28, 28, device=device)
out = model(x)
if device.type == "mps":
    torch.mps.synchronize()
```

### 2. Apple MLX (Python / Native Metal)
```python
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

model = StandardCNN()
model.eval()
mx.eval(model.parameters())
compiled_model = mx.compile(model)
x = mx.random.normal((128, 28, 28, 1))
out = compiled_model(x)
mx.eval(out)
```

### 3. Hugging Face Candle (Rust / Metal)
```rust
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2dConfig, Linear, Module,
    ModuleT, VarBuilder, VarMap,
};

struct StandardCNN {
    conv1: candle_nn::Conv2d,
    bn1: BatchNorm,
    conv2: candle_nn::Conv2d,
    bn2: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl StandardCNN {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(1, 32, 3, conv_cfg, vs.pp("conv1"))?;
        let bn1 = batch_norm(32, BatchNormConfig::default(), vs.pp("bn1"))?;
        let conv2 = conv2d(32, 64, 3, conv_cfg, vs.pp("conv2"))?;
        let bn2 = batch_norm(64, BatchNormConfig::default(), vs.pp("bn2"))?;
        let fc1 = linear(64 * 7 * 7, 128, vs.pp("fc1"))?;
        let fc2 = linear(128, 10, vs.pp("fc2"))?;
        Ok(Self { conv1, bn1, conv2, bn2, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.bn1.forward_t(&self.conv1.forward(xs)?, false)?.relu()?;
        let xs = xs.max_pool2d(2)?; // (B, 32, 14, 14)
        let xs = self.bn2.forward_t(&self.conv2.forward(&xs)?, false)?.relu()?;
        let xs = xs.max_pool2d(2)?; // (B, 64, 7, 7)
        let (b, _, _, _) = xs.dims4()?;
        let xs = xs.reshape((b, 64 * 7 * 7))?;
        let xs = self.fc1.forward(&xs)?.relu()?;
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}

// Execution snippet:
let device = Device::new_metal(0)?;
let varmap = VarMap::new();
let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
let model = StandardCNN::new(vs)?;
let x = Tensor::randn(0.0f32, 1.0f32, (128, 1, 28, 28), &device)?;
let out = model.forward(&x)?;
device.synchronize()?;
```

---

## How to Reproduce

### 1. Prerequisites
Ensure you have Python 3.13 and Rust 1.99.1 (rustup or Homebrew) installed:
```bash
# Install Rust (if not installed)
brew install rust
```

### 2. Clone & Set Up Environment
```bash
git clone https://github.com/adityavsb/pytorch-vs-mlx-vs-candle.git
cd pytorch-vs-mlx-vs-candle

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build the Rust Candle Binary
```bash
cd rust_candle
cargo build --release
cd ..
```

### 4. Run the Full 3-Way Shootout
```bash
python benchmark_all.py
```
This will automatically execute PyTorch, Apple MLX, and Candle (Rust) over 5 trials, display the comparison table, export `results/latest.json`, and generate `assets/shootout_results.png`.

### 5. Quickstart with Make
For convenience, common engineering workflows are wrapped in the project `Makefile`:
```bash
# Run unit tests (Python & Rust shape parity)
make test

# Validate that README summary table matches results/latest.json
make check

# Build release binary and run full 5-trial shootout
make bench
```

---

## References

- A. Hannun, J. Digani, A. Katharopoulos, R. Collobert. *MLX: Efficient and flexible machine learning on Apple silicon*, 2023. https://github.com/ml-explore/mlx
- A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS 2019. arXiv:1912.01703
- PyTorch MPS backend documentation. https://pytorch.org/docs/stable/notes/mps.html
- Hugging Face Candle. https://github.com/huggingface/candle
- Apple, *Metal Performance Shaders Graph*. https://developer.apple.com/documentation/metalperformanceshadersgraph
- Apple, *Mac mini (2024) – Tech Specs*. https://support.apple.com/en-us/121555

---

## License

This project is licensed under the [MIT License](LICENSE).
