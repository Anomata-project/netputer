// SPDX-License-Identifier: Apache-2.0
use assert_cmd::Command;
use npf::Network;
use std::path::{Path, PathBuf};

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "npf-toolkit-it-{}-{}-{}",
        tag,
        std::process::id(),
        // Per-test nanosecond suffix so parallel cases don't collide.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn gen(preset: &str, out_path: &Path) {
    Command::cargo_bin("npf-gen")
        .unwrap()
        .arg(preset)
        .arg("--output")
        .arg(out_path)
        .assert()
        .success();
}

fn round_trip_preset(preset: &str) {
    let dir = unique_tmp_dir(preset);
    let out = dir.join(format!("{preset}.npf"));
    gen(preset, &out);

    let bytes = std::fs::read(&out).expect("read generated file");
    let parsed = Network::parse(&bytes).expect("parse");
    let reserialized = parsed.to_bytes();

    assert_eq!(
        bytes, reserialized,
        "byte-for-byte round-trip failed for preset {preset}"
    );
}

#[test]
fn round_trip_tiny() {
    round_trip_preset("tiny");
}

#[test]
fn round_trip_mlp() {
    round_trip_preset("mlp");
}

#[test]
fn round_trip_lenet5() {
    round_trip_preset("lenet5");
}

fn validate_preset(preset: &str) {
    let dir = unique_tmp_dir(preset);
    let out = dir.join(format!("{preset}.npf"));
    gen(preset, &out);

    Command::cargo_bin("npf-validate")
        .unwrap()
        .arg(&out)
        .assert()
        .success();
}

#[test]
fn validate_tiny() {
    validate_preset("tiny");
}

#[test]
fn validate_mlp() {
    validate_preset("mlp");
}

#[test]
fn validate_lenet5() {
    validate_preset("lenet5");
}

#[test]
fn corrupted_tiny_fails_validation() {
    let dir = unique_tmp_dir("corrupt");
    let out = dir.join("tiny.npf");
    gen("tiny", &out);

    // Tiny layout: header (60 + name_len 4) = 64 bytes, then Dense record
    // (8 preamble + 8 params = 16) + ReLU record (8 preamble + 0 params = 8).
    // Weight section begins at offset 88.
    let mut bytes = std::fs::read(&out).expect("read");
    bytes[88] ^= 0xFF;
    std::fs::write(&out, &bytes).expect("overwrite corrupted");

    let output = Command::cargo_bin("npf-validate")
        .unwrap()
        .arg(&out)
        .assert()
        .failure();

    let stderr = String::from_utf8_lossy(&output.get_output().stderr).into_owned();
    assert!(
        stderr.contains("CRC"),
        "expected CRC error on corrupted file, got stderr: {stderr}"
    );
}
