// SPDX-License-Identifier: Apache-2.0
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::fmt;

const INPUT_FEATURES: usize = 2;
const HIDDEN_FEATURES: usize = 4;
const OUTPUT_FEATURES: usize = 1;
const RNG_SEED: u64 = 42;
const LEARNING_RATE: f32 = 0.1;
const MAX_EPOCHS: usize = 10_000;
const EARLY_STOP_MSE: f32 = 0.01;
const LOG_INTERVAL: usize = 500;
const XOR_DATASET: [([f32; 2], f32); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

#[derive(Debug, Clone, PartialEq)]
struct Params {
    dense1_weights: Vec<f32>,
    dense1_biases: Vec<f32>,
    dense2_weights: Vec<f32>,
    dense2_biases: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct ForwardCache {
    hidden_pre: Vec<f32>,
    hidden: Vec<f32>,
    output_pre: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct TrainingSummary {
    final_epoch: usize,
    final_loss: f32,
    stopped_early: bool,
}

impl fmt::Display for TrainingSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "epoch {} loss {:.6}{}",
            self.final_epoch,
            self.final_loss,
            if self.stopped_early {
                " (early stop)"
            } else {
                ""
            }
        )
    }
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

fn forward_cache(input: &[f32; 2], params: &Params) -> ForwardCache {
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
    let output = sigmoid_forward(&output_pre);

    ForwardCache {
        hidden_pre,
        hidden,
        output_pre,
        output,
    }
}

fn network_forward(input: &[f32; 2], params: &Params) -> f32 {
    forward_cache(input, params).output[0]
}

fn example_loss(output: f32, target: f32) -> f32 {
    let diff = output - target;
    diff * diff
}

fn mean_squared_error(params: &Params) -> f32 {
    let total_loss: f32 = XOR_DATASET
        .iter()
        .map(|(input, target)| example_loss(network_forward(input, params), *target))
        .sum();
    total_loss / XOR_DATASET.len() as f32
}

fn train_example(params: &mut Params, input: &[f32; 2], target: f32, learning_rate: f32) {
    let cache = forward_cache(input, params);
    let out = cache.output[0];

    // d_loss/d_out for mean squared error on a single scalar output.
    let d_loss_d_out = 2.0 * (out - target);
    // d_out/d_z2 for sigmoid.
    let delta2 = d_loss_d_out * out * (1.0 - out);

    let mut grad_dense2_weights = vec![0.0; params.dense2_weights.len()];
    let mut grad_dense2_biases = vec![0.0; params.dense2_biases.len()];
    for hidden_idx in 0..HIDDEN_FEATURES {
        grad_dense2_weights[hidden_idx] = delta2 * cache.hidden[hidden_idx];
    }
    grad_dense2_biases[0] = delta2;

    let mut hidden_delta = vec![0.0; HIDDEN_FEATURES];
    for hidden_idx in 0..HIDDEN_FEATURES {
        let d_loss_d_hidden = params.dense2_weights[hidden_idx] * delta2;
        // d_tanh/d_z1 = 1 - h^2, using the activated hidden value h.
        hidden_delta[hidden_idx] =
            d_loss_d_hidden * (1.0 - cache.hidden[hidden_idx] * cache.hidden[hidden_idx]);
    }

    let mut grad_dense1_weights = vec![0.0; params.dense1_weights.len()];
    let mut grad_dense1_biases = vec![0.0; params.dense1_biases.len()];
    for hidden_idx in 0..HIDDEN_FEATURES {
        for input_idx in 0..INPUT_FEATURES {
            grad_dense1_weights[hidden_idx * INPUT_FEATURES + input_idx] =
                hidden_delta[hidden_idx] * input[input_idx];
        }
        grad_dense1_biases[hidden_idx] = hidden_delta[hidden_idx];
    }

    for (weight, grad) in params
        .dense2_weights
        .iter_mut()
        .zip(grad_dense2_weights.iter())
    {
        *weight -= learning_rate * grad;
    }
    for (bias, grad) in params
        .dense2_biases
        .iter_mut()
        .zip(grad_dense2_biases.iter())
    {
        *bias -= learning_rate * grad;
    }
    for (weight, grad) in params
        .dense1_weights
        .iter_mut()
        .zip(grad_dense1_weights.iter())
    {
        *weight -= learning_rate * grad;
    }
    for (bias, grad) in params
        .dense1_biases
        .iter_mut()
        .zip(grad_dense1_biases.iter())
    {
        *bias -= learning_rate * grad;
    }
}

fn train_xor(params: &mut Params, rng: &mut StdRng) -> TrainingSummary {
    let mut training_order = [0usize, 1, 2, 3];

    for epoch in 1..=MAX_EPOCHS {
        training_order.shuffle(rng);
        for example_idx in training_order {
            let (input, target) = XOR_DATASET[example_idx];
            train_example(params, &input, target, LEARNING_RATE);
        }

        let loss = mean_squared_error(params);
        if epoch % LOG_INTERVAL == 0 {
            println!("epoch {epoch:>5} loss {loss:.6}");
        }
        if loss < EARLY_STOP_MSE {
            println!("epoch {epoch:>5} loss {loss:.6} (stopping)");
            return TrainingSummary {
                final_epoch: epoch,
                final_loss: loss,
                stopped_early: true,
            };
        }
    }

    let loss = mean_squared_error(params);
    println!("epoch {:>5} loss {:.6} (max epochs)", MAX_EPOCHS, loss);
    TrainingSummary {
        final_epoch: MAX_EPOCHS,
        final_loss: loss,
        stopped_early: false,
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let mut params = init_params(&mut rng);
    let summary = train_xor(&mut params, &mut rng);
    println!(
        "trained xor network ({} weights, {} biases) {}",
        params.dense1_weights.len() + params.dense2_weights.len(),
        params.dense1_biases.len() + params.dense2_biases.len()
        ,
        summary
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

    #[test]
    fn training_reduces_loss() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let mut params = init_params(&mut rng);
        let initial_loss = mean_squared_error(&params);

        for _ in 0..50 {
            let mut training_order = [0usize, 1, 2, 3];
            training_order.shuffle(&mut rng);
            for example_idx in training_order {
                let (input, target) = XOR_DATASET[example_idx];
                train_example(&mut params, &input, target, LEARNING_RATE);
            }
        }

        let final_loss = mean_squared_error(&params);
        assert!(final_loss < initial_loss);
    }
}
