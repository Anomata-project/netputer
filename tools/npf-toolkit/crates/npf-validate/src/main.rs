// SPDX-License-Identifier: Apache-2.0
use clap::Parser;
use npf::Network;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "npf-validate",
    about = "Validate a Netputer Package Format (.npf) file."
)]
struct Args {
    /// Path to the .npf file to validate.
    file: PathBuf,
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{}KB", (bytes + KB / 2) / KB)
    } else {
        format!("{bytes}B")
    }
}

fn run(args: Args) -> Result<(), String> {
    let raw = std::fs::read(&args.file)
        .map_err(|e| format!("could not read {}: {e}", args.file.display()))?;

    let net = Network::parse(&raw).map_err(|e| e.to_string())?;

    let filename = args
        .file
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.file.display().to_string());

    let param_count = net.total_params();
    let size = format_size(raw.len() as u64);

    println!("OK: {filename} ({param_count} params, {size})");
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
