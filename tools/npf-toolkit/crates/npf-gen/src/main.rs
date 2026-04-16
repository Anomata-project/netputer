// SPDX-License-Identifier: Apache-2.0
use clap::{Parser, ValueEnum};
use npf::{Header, Layer, Network, PaddingMode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preset {
    Tiny,
    Mlp,
    Lenet5,
}

impl Preset {
    fn default_filename(self) -> &'static str {
        match self {
            Preset::Tiny => "tiny.npf",
            Preset::Mlp => "mlp.npf",
            Preset::Lenet5 => "lenet5.npf",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "npf-gen",
    about = "Generate example Netputer Package Format (.npf) files."
)]
struct Args {
    /// Which preset to generate.
    preset: Preset,

    /// Output path. Defaults to <preset>.npf in the current directory.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn build_tiny() -> Network {
    let header = Header::new("tiny", [2, 0, 0, 0], [1, 0, 0, 0]);
    let layers = vec![
        Layer::Dense {
            in_features: 2,
            out_features: 1,
        },
        Layer::ReLU,
    ];
    fill(header, layers)
}

fn build_mlp() -> Network {
    let header = Header::new("mlp", [4, 0, 0, 0], [2, 0, 0, 0]);
    let layers = vec![
        Layer::Dense {
            in_features: 4,
            out_features: 3,
        },
        Layer::ReLU,
        Layer::Dense {
            in_features: 3,
            out_features: 2,
        },
        Layer::Softmax { axis: 0 },
    ];
    fill(header, layers)
}

fn build_lenet5() -> Network {
    let header = Header::new("lenet5", [1, 28, 28, 0], [10, 0, 0, 0]);
    let layers = vec![
        Layer::Conv2D {
            in_channels: 1,
            out_channels: 6,
            kernel_h: 5,
            kernel_w: 5,
            stride_h: 1,
            stride_w: 1,
            padding_mode: PaddingMode::Same,
        },
        Layer::ReLU,
        Layer::MaxPool2D {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        },
        Layer::Conv2D {
            in_channels: 6,
            out_channels: 16,
            kernel_h: 5,
            kernel_w: 5,
            stride_h: 1,
            stride_w: 1,
            padding_mode: PaddingMode::Valid,
        },
        Layer::ReLU,
        Layer::MaxPool2D {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        },
        Layer::Flatten,
        Layer::Dense {
            in_features: 400,
            out_features: 120,
        },
        Layer::ReLU,
        Layer::Dense {
            in_features: 120,
            out_features: 84,
        },
        Layer::ReLU,
        Layer::Dense {
            in_features: 84,
            out_features: 10,
        },
        Layer::Softmax { axis: 0 },
    ];
    fill(header, layers)
}

fn fill(header: Header, layers: Vec<Layer>) -> Network {
    // Deterministic RNG so output is reproducible. Values are not meaningful —
    // just structurally valid float32s.
    let mut rng = StdRng::seed_from_u64(42);
    let weight_count: usize = layers.iter().map(Layer::weight_count).sum();
    let bias_count: usize = layers.iter().map(Layer::bias_count).sum();

    let weights: Vec<f32> = (0..weight_count)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();
    let biases: Vec<f32> = (0..bias_count)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    Network {
        header,
        layers,
        weights,
        biases,
    }
}

fn run(args: Args) -> Result<(), String> {
    let net = match args.preset {
        Preset::Tiny => build_tiny(),
        Preset::Mlp => build_mlp(),
        Preset::Lenet5 => build_lenet5(),
    };

    let out_path = args
        .output
        .unwrap_or_else(|| PathBuf::from(args.preset.default_filename()));

    let bytes = net
        .to_bytes()
        .map_err(|e| format!("could not serialize {}: {e}", out_path.display()))?;
    std::fs::write(&out_path, &bytes)
        .map_err(|e| format!("could not write {}: {e}", out_path.display()))?;

    println!(
        "wrote {} ({} bytes, {} params)",
        out_path.display(),
        bytes.len(),
        net.total_params()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("ERROR: {msg}");
            ExitCode::FAILURE
        }
    }
}
