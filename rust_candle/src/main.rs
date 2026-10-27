use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2dConfig, Linear, Module,
    ModuleT, VarBuilder, VarMap,
};
use std::time::Instant;

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
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            fc1,
            fc2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, 1, 28, 28); BatchNorm in inference mode (train = false) to match
        // torch `model.eval()` and mlx `model.eval()`.
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

fn sync_device(device: &Device) -> Result<()> {
    // Wait for all queued Metal command buffers; no host copy (matches
    // torch.mps.synchronize() and mx.eval()).
    device.synchronize()?;
    Ok(())
}

// Linear interpolation, identical to numpy.percentile(method="linear").
fn pct(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn main() -> Result<()> {
    // 1. Initialize Apple Silicon Metal Device
    let device = match Device::new_metal(0) {
        Ok(d) => {
            println!("[Hugging Face Candle] Metal GPU device initialized: {:?}", d);
            d
        }
        Err(e) => {
            eprintln!("[Hugging Face Candle] Metal init failed ({}), falling back to CPU", e);
            Device::Cpu
        }
    };

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let model = StandardCNN::new(vs)?;

    let batch_sizes = vec![32, 64, 128, 256, 512];
    let min_seconds = 2.0f64;
    let min_iters = 50usize;

    println!("\n>>> Candle (Rust) Batch Throughput Benchmark <<<");
    let mut throughput_map = serde_json::Map::new();

    for &bs in &batch_sizes {
        let x = Tensor::randn(0.0f32, 1.0f32, (bs, 1, 28, 28), &device)?;

        // Warmup
        for _ in 0..5 {
            let out = model.forward(&x)?;
            let _ = out;
            sync_device(&device)?;
        }

        let mut n = 0usize;
        let start = Instant::now();
        loop {
            let out = model.forward(&x)?;
            let _ = out;
            sync_device(&device)?;
            n += 1;
            if n >= min_iters && start.elapsed().as_secs_f64() >= min_seconds {
                break;
            }
        }
        let duration = start.elapsed().as_secs_f64();

        let total_imgs = (bs * n) as f64;
        let img_per_sec = total_imgs / duration;
        throughput_map.insert(bs.to_string(), serde_json::Value::from(img_per_sec));

        println!(
            "  Batch {:3}: {:8.0} img/s ({:6.2} ms/step, {} iters in {:.2}s)",
            bs,
            img_per_sec,
            (duration / n as f64) * 1000.0,
            n,
            duration
        );
    }

    println!("\n>>> Candle (Rust) Batch-1 Latency Benchmark (500 runs) <<<");
    let x1 = Tensor::randn(0.0f32, 1.0f32, (1, 1, 28, 28), &device)?;

    // Warmup
    for _ in 0..20 {
        let out = model.forward(&x1)?;
        let _ = out;
        sync_device(&device)?;
    }

    let mut latencies_us = Vec::with_capacity(500);
    for _ in 0..500 {
        let t0 = Instant::now();
        let out = model.forward(&x1)?;
        let _ = out;
        sync_device(&device)?;
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = pct(&latencies_us, 0.50);
    let p95 = pct(&latencies_us, 0.95);
    let p99 = pct(&latencies_us, 0.99);

    println!(
        "[Candle Batch-1 Latency] P50: {:6.1} μs | P95: {:6.1} μs | P99: {:6.1} μs",
        p50, p95, p99
    );

    // Output JSON summary line for automated benchmark orchestrators
    let json_output = serde_json::json!({
        "engine": "Candle (Rust Metal)",
        "throughput": throughput_map,
        "latency_us": {
            "p50": p50,
            "p95": p95,
            "p99": p99
        }
    });
    println!("JSON_RESULT: {}", json_output);

    Ok(())
}
