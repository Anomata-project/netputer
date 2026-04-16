// SPDX-License-Identifier: Apache-2.0
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const INPUT_FEATURES: usize = 2;
const HIDDEN_FEATURES: usize = 4;
const OUTPUT_FEATURES: usize = 1;
const RNG_SEED: u64 = 42;

#[derive(Debug, Clone, PartialEq)]
struct Params {
    dense1_weights: Vec<f32>,
    dense1_biases: Vec<f32>,
    dense2_weights: Vec<f32>,
    dense2_biases: Vec<f32>,
}

fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0f32 / (fan_in + fan_out) as f32).sqrt()
}

fn init_dense_weights(
    rng: &mut StdRng,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let limit = xavier_limit(in_features, out_features);
    let mut weights = Vec::with_capacity(in_features * out_features);

    // Dense weights follow the NPF Dense convention used by this crate:
    // w[out_idx * in_features + in_idx].
    for _out_idx in 0..out_features {
        for _in_idx in 0..in_features {
            weights.push(rng.gen_range(-limit..=limit));
        }
    }

    weights
}

fn init_dense_biases(out_features: usize) -> Vec<f32> {
    vec![0.0; out_features]
}

fn init_params(rng: &mut StdRng) -> Params {
    Params {
        dense1_weights: init_dense_weights(rng, INPUT_FEATURES, HIDDEN_FEATURES),
        dense1_biases: init_dense_biases(HIDDEN_FEATURES),
        dense2_weights: init_dense_weights(rng, HIDDEN_FEATURES, OUTPUT_FEATURES),
        dense2_biases: init_dense_biases(OUTPUT_FEATURES),
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let params = init_params(&mut rng);
    println!(
        "initialized xor network ({} weights, {} biases)",
        params.dense1_weights.len() + params.dense2_weights.len(),
        params.dense1_biases.len() + params.dense2_biases.len()
    );
}
