# Apple Silicon Deep Learning Shootout: PyTorch vs. Apple MLX vs. Hugging Face Candle (Rust)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.99.1-orange.svg)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(MPS)-EE4C2C.svg)](https://pytorch.org/)
[![Apple MLX](https://img.shields.io/badge/Apple%20MLX-0.32%20(Native%20GPU)-0071E3.svg)](https://github.com/ml-explore/mlx)
[![Candle](https://img.shields.io/badge/HF%20Candle-0.11%20(Rust%20Metal)-yellow.svg)](https://github.com/huggingface/candle)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20M4%20Mac%20mini-black.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An empirical hardware shootout testing three distinct deep learning paradigms head-to-head on the **Apple M4 Mac mini**:

1. **PyTorch (Python / MPS):** The battle-tested industry standard using Apple's Metal Performance Shaders backend.
2. **Apple MLX (Python / C++ Metal):** Apple ML Research's framework engineered from scratch for unified memory zero-copy operations.
3. **Hugging Face Candle (Rust / Metal):** A minimalist systems-level framework compiled in pure Rust, eliminating Python's GIL and runtime garbage collection.

---

## Hardware & Testbench Specifications

```text
Machine:             Apple Mac mini (Late 2024)
SoC:                 Apple M4
Unified Memory:      16.00 GB (Shared CPU/GPU)
Platform / OS:       macOS Darwin (arm64)
Rust Toolchain:      rustc 1.99.1 / cargo 1.99.1
Python Runtime:      Python 3.13.6
PyTorch Backend:     torch 2.14.1 (MPS - Metal Performance Shaders)
Apple MLX Backend:   mlx 0.32.4 (Device: gpu, 0)
HF Candle Backend:   candle-core 0.11.0 (Metal Device: 1)
```

---

## Benchmark Results

All three frameworks were tested on identical Deep Convolutional Neural Network architectures ($2\times$ `Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool` $\rightarrow$ `Dense`):

### 1. Throughput & Latency Summary

| Metric | PyTorch (MPS) | Apple MLX (Native) | Candle (Rust Metal) | Winner / Insight |
| :--- | :---: | :---: | :---: | :--- |
| **Batch 32 Throughput** | `40,482 img/s` | `35,017 img/s` | `6,421 img/s` | **PyTorch** (+15% vs MLX) |
| **Batch 64 Throughput** | `55,941 img/s` | `43,707 img/s` | `6,429 img/s` | **PyTorch** (+28% vs MLX) |
| **Batch 128 Throughput** | `60,256 img/s` | `47,093 img/s` | `5,990 img/s` | **PyTorch** (+28% vs MLX) |
| **Batch 256 Throughput** | `60,919 img/s` | `49,954 img/s` | `5,812 img/s` | **PyTorch** (+22% vs MLX) |
| **Batch 512 Throughput** | **`62,229 img/s`** | `52,903 img/s` | `5,458 img/s` | **PyTorch** (+17% vs MLX) 🚀 |
| **Batch-1 Latency (P50 Median)** | `484.8 μs` | **`297.3 μs`** | `915.0 μs` | **Apple MLX is 1.63x faster** ⚡ |
| **Batch-1 Latency (P95 Tail)** | `564.5 μs` | **`317.5 μs`** | `981.5 μs` | **Apple MLX is 1.78x faster** ⚡ |
| **Batch-1 Latency (P99 Max)** | `623.3 μs` | **`382.2 μs`** | `1041.0 μs` | **Apple MLX is 1.63x faster** ⚡ |
| **Runtime Overhead** | Python GIL + C++ | Python + C++ Metal | **Zero GC / Pure Binary** | **Candle (Rust)** 🛡️ |

![Shootout Results](assets/shootout_results.png)

---

## Key Systems Engineering Takeaways

### 1. Apple MLX is the King of Real-Time Microsecond Latency
* At **Batch Size = 1** (real-time camera feeds, edge robotics, audio streams), **Apple MLX crushed both PyTorch and Candle**, achieving a **P50 median latency of only 297 microseconds** ($0.29$ ms).
* **Why MLX Wins at Batch 1:** MLX’s memory allocator is designed specifically for Apple Silicon's unified memory. Unlike PyTorch (which incurs command buffer synchronization latency over Metal), MLX’s lazy evaluation graph (`@mx.compile`) dispatches lightweight Metal compute commands with near-zero driver overhead.

### 2. PyTorch MPS Dominates High-Volume Throughput
* When processing large batches ($128$ to $512$), **PyTorch (MPS) peaked at over 62,200 images/second**, beating MLX by ~17–28%.
* **Why PyTorch Wins at High Batches:** PyTorch’s MPS backend relies on Apple's highly mature `MPSGraph` library, which contains hand-tuned GEMM and convolution shaders specifically optimized for Apple Silicon SIMD execution units when matrix dimensions are large.

### 3. Hugging Face Candle in Rust: The Zero-Python Future
* Candle in Rust demonstrated rock-solid predictability with **zero garbage collection pauses**, compiling into a single standalone binary.
* While Candle's custom Metal convolution kernels in version `0.11` are not yet as heavily vector-tuned as Apple's proprietary `MPSGraph` library, its ability to run neural networks natively in compiled Rust without a Python interpreter makes it ideal for embedded and systems-level production deployments.

---

## Code Comparison Across Frameworks

### 1. PyTorch (Python / MPS)
```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 14 * 14, 10)
    def forward(self, x):
        return self.fc(torch.flatten(torch.relu(self.conv(x)), 1))

model = CNN().to("mps")
out = model(torch.randn(128, 1, 28, 28, device="mps"))
torch.mps.synchronize()
```

### 2. Apple MLX (Python / Native Metal)
```python
import mlx.core as mx
import mlx.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 14 * 14, 10)
    def __call__(self, x):
        return self.fc(self.conv(x).reshape((x.shape[0], -1)))

model = mx.compile(CNN())
out = model(mx.random.normal((128, 28, 28, 1)))
mx.eval(out)  # Force zero-copy GPU evaluation
```

### 3. Hugging Face Candle (Rust / Metal)
```rust
use candle_core::{Device, Tensor};
use candle_nn::{conv2d, linear, Conv2dConfig, Module, VarBuilder, VarMap};

let device = Device::new_metal(0)?;
let vs = VarBuilder::from_varmap(&VarMap::new(), candle_core::DType::F32, &device);
let conv = conv2d(1, 32, 3, Conv2dConfig { padding: 1, ..Default::default() }, vs.pp("c"))?;

let x = Tensor::randn(0.0f32, 1.0f32, (128, 1, 28, 28), &device)?;
let out = conv.forward(&x)?;
```

---

## How to Reproduce

### 1. Prerequisites
Ensure you have Python 3.11+ and the Rust toolchain installed:
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
This will automatically execute PyTorch, Apple MLX, and Candle (Rust), display the comparison table, and generate `assets/shootout_results.png`.

---

## License

This project is licensed under the [MIT License](LICENSE).
