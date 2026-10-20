use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{conv2d, linear, Conv2dConfig, Linear, Module, VarBuilder, VarMap};
use std::time::Instant;

struct StandardCNN {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
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
        let conv2 = conv2d(32, 64, 3, conv_cfg, vs.pp("conv2"))?;
        let fc1 = linear(64 * 7 * 7, 128, vs.pp("fc1"))?;
        let fc2 = linear(128, 10, vs.pp("fc2"))?;
        Ok(Self { conv1, conv2, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (B, 1, 28, 28)
        let xs = self.conv1.forward(xs)?.relu()?;
        let xs = xs.max_pool2d(2)?; // (B, 32, 14, 14)
        let xs = self.conv2.forward(&xs)?.relu()?;
        let xs = xs.max_pool2d(2)?; // (B, 64, 7, 7)
        let (b, _, _, _) = xs.dims4()?;
        let xs = xs.reshape((b, 64 * 7 * 7))?;
        let xs = self.fc1.forward(&xs)?.relu()?;
        let xs = self.fc2.forward(&xs)?;
        Ok(xs)
    }
}

fn sync_device(t: &Tensor) -> Result<()> {
    // Reading data from the tensor forces Apple Metal GPU synchronization
    let _ = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(())
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
    let iterations = 40;

    println!("\n>>> Candle (Rust) Batch Throughput Benchmark <<<");
    let mut throughput_map = serde_json::Map::new();

    for &bs in &batch_sizes {
        let x = Tensor::randn(0.0f32, 1.0f32, (bs, 1, 28, 28), &device)?;

        // Warmup
        for _ in 0..5 {
            let out = model.forward(&x)?;
            sync_device(&out)?;
        }

        let start = Instant::now();
        for _ in 0..iterations {
            let out = model.forward(&x)?;
            sync_device(&out)?;
        }
        let duration = start.elapsed().as_secs_f64();

        let total_imgs = (bs * iterations) as f64;
        let img_per_sec = total_imgs / duration;
        throughput_map.insert(bs.to_string(), serde_json::Value::from(img_per_sec));

        println!(
            "  Batch {:3}: {:8.0} img/s ({:6.2} ms/step)",
            bs,
            img_per_sec,
            (duration / iterations as f64) * 1000.0
        );
    }

    println!("\n>>> Candle (Rust) Batch-1 Latency Benchmark (500 runs) <<<");
    let x1 = Tensor::randn(0.0f32, 1.0f32, (1, 1, 28, 28), &device)?;

    // Warmup
    for _ in 0..20 {
        let out = model.forward(&x1)?;
        sync_device(&out)?;
    }

    let mut latencies_us = Vec::with_capacity(500);
    for _ in 0..500 {
        let t0 = Instant::now();
        let out = model.forward(&x1)?;
        sync_device(&out)?;
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[(latencies_us.len() as f64 * 0.50) as usize];
    let p95 = latencies_us[(latencies_us.len() as f64 * 0.95) as usize];
    let p99 = latencies_us[(latencies_us.len() as f64 * 0.99) as usize];

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
