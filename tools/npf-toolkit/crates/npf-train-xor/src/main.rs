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

fn dense_forward(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_features);
    assert_eq!(weights.len(), in_features * out_features);
    assert_eq!(biases.len(), out_features);

    let mut output = vec![0.0; out_features];
    for out_idx in 0..out_features {
        let mut sum = biases[out_idx];
        for in_idx in 0..in_features {
            sum += weights[out_idx * in_features + in_idx] * input[in_idx];
        }
        output[out_idx] = sum;
    }
    output
}

fn tanh_forward(input: &[f32]) -> Vec<f32> {
    input.iter().map(|value| value.tanh()).collect()
}

fn sigmoid_forward(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect()
}

fn network_forward(input: &[f32; 2], params: &Params) -> f32 {
    let hidden_pre = dense_forward(
        input,
        &params.dense1_weights,
        &params.dense1_biases,
        INPUT_FEATURES,
        HIDDEN_FEATURES,
    );
    let hidden = tanh_forward(&hidden_pre);
    let output_pre = dense_forward(
        &hidden,
        &params.dense2_weights,
        &params.dense2_biases,
        HIDDEN_FEATURES,
        OUTPUT_FEATURES,
    );
    sigmoid_forward(&output_pre)[0]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_pass_matches_hand_computed_output() {
        let params = Params {
            dense1_weights: vec![0.2, -0.4, 0.7, 0.1, -0.3, 0.8, 0.5, -0.6],
            dense1_biases: vec![0.1, -0.2, 0.05, 0.3],
            dense2_weights: vec![0.4, -0.7, 0.2, 0.9],
            dense2_biases: vec![-0.15],
        };

        let actual = network_forward(&[0.25, -0.5], &params);

        let z1_0: f32 = 0.1 + 0.2 * 0.25 + (-0.4 * -0.5);
        let z1_1: f32 = -0.2 + 0.7 * 0.25 + 0.1 * -0.5;
        let z1_2: f32 = 0.05 + -0.3 * 0.25 + 0.8 * -0.5;
        let z1_3: f32 = 0.3 + 0.5 * 0.25 + -0.6 * -0.5;
        let h0 = z1_0.tanh();
        let h1 = z1_1.tanh();
        let h2 = z1_2.tanh();
        let h3 = z1_3.tanh();
        let z2: f32 = -0.15 + 0.4 * h0 + -0.7 * h1 + 0.2 * h2 + 0.9 * h3;
        let expected: f32 = 1.0 / (1.0 + (-z2).exp());

        assert!((actual - expected).abs() < 1e-6);
    }
}
