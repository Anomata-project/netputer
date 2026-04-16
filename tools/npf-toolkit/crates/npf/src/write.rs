// SPDX-License-Identifier: Apache-2.0
use crate::error::{NpfError, Result};
use crate::types::{Layer, Network, PaddingMode, ENDIAN_LE, MAGIC, PRECISION_F32, VERSION};

fn write_layer(out: &mut Vec<u8>, layer: &Layer) {
    out.extend_from_slice(&layer.layer_type().tag().to_le_bytes());
    out.extend_from_slice(&layer.param_bytes().to_le_bytes());

    match layer {
        Layer::Dense {
            in_features,
            out_features,
        } => {
            out.extend_from_slice(&in_features.to_le_bytes());
            out.extend_from_slice(&out_features.to_le_bytes());
        }
        Layer::Conv2D {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_mode,
        } => {
            out.extend_from_slice(&in_channels.to_le_bytes());
            out.extend_from_slice(&out_channels.to_le_bytes());
            out.extend_from_slice(&kernel_h.to_le_bytes());
            out.extend_from_slice(&kernel_w.to_le_bytes());
            out.extend_from_slice(&stride_h.to_le_bytes());
            out.extend_from_slice(&stride_w.to_le_bytes());
            let pad = match padding_mode {
                PaddingMode::Valid => 0u32,
                PaddingMode::Same => 1u32,
            };
            out.extend_from_slice(&pad.to_le_bytes());
        }
        Layer::MaxPool2D {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        } => {
            out.extend_from_slice(&kernel_h.to_le_bytes());
            out.extend_from_slice(&kernel_w.to_le_bytes());
            out.extend_from_slice(&stride_h.to_le_bytes());
            out.extend_from_slice(&stride_w.to_le_bytes());
        }
        Layer::Softmax { axis } => {
            out.extend_from_slice(&axis.to_le_bytes());
        }
        Layer::Flatten | Layer::ReLU | Layer::Tanh | Layer::Sigmoid => {}
    }
}

impl Network {
    /// Serialize this network into a fresh byte vector. The CRC32 stored in the
    /// header is computed here; the caller does not need to pre-populate it.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let expected_weight_count: usize = self.layers.iter().map(Layer::weight_count).sum();
        if self.weights.len() != expected_weight_count {
            return Err(NpfError::WeightCountMismatch {
                expected: expected_weight_count,
                actual: self.weights.len(),
            });
        }

        let expected_bias_count: usize = self.layers.iter().map(Layer::bias_count).sum();
        if self.biases.len() != expected_bias_count {
            return Err(NpfError::BiasCountMismatch {
                expected: expected_bias_count,
                actual: self.biases.len(),
            });
        }

        let mut out = Vec::new();

        // Weight bytes first, so we can CRC them.
        let mut weight_bytes = Vec::with_capacity(self.weights.len() * 4);
        for w in &self.weights {
            weight_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let crc = crc32fast::hash(&weight_bytes);

        // ---- Header ----
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&ENDIAN_LE.to_le_bytes());
        out.extend_from_slice(&PRECISION_F32.to_le_bytes());
        out.extend_from_slice(&crc.to_le_bytes());

        let name_bytes = self.header.name.as_bytes();
        out.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(name_bytes);

        for d in self.header.input_shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for d in self.header.output_shape {
            out.extend_from_slice(&d.to_le_bytes());
        }

        let layer_count = self.layers.len() as u32;
        out.extend_from_slice(&layer_count.to_le_bytes());

        // ---- Architecture ----
        for layer in &self.layers {
            write_layer(&mut out, layer);
        }

        // ---- Weights ----
        out.extend_from_slice(&weight_bytes);

        // ---- Biases ----
        for b in &self.biases {
            out.extend_from_slice(&b.to_le_bytes());
        }

        Ok(out)
    }

    pub fn write_to<W: std::io::Write>(&self, mut w: W) -> Result<()> {
        let bytes = self.to_bytes()?;
        w.write_all(&bytes)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Header, Layer};

    #[test]
    fn round_trip_hand_built_network() {
        let header = Header::new("round", [2, 0, 0, 0], [1, 0, 0, 0]);
        let net = Network {
            header,
            layers: vec![
                Layer::Dense {
                    in_features: 2,
                    out_features: 1,
                },
                Layer::ReLU,
            ],
            weights: vec![0.25, -0.5],
            biases: vec![0.125],
        };

        let bytes = net.to_bytes().expect("serialize valid network");
        let parsed = Network::parse(&bytes).expect("round-trip parse");

        assert_eq!(parsed.header.name, "round");
        assert_eq!(parsed.layers, net.layers);
        assert_eq!(parsed.weights, net.weights);
        assert_eq!(parsed.biases, net.biases);

        // Re-serialize and confirm byte-for-byte identity
        let bytes2 = parsed.to_bytes().expect("re-serialize parsed network");
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn rejects_too_few_weights() {
        let net = Network {
            header: Header::new("few-weights", [2, 0, 0, 0], [1, 0, 0, 0]),
            layers: vec![Layer::Dense {
                in_features: 2,
                out_features: 1,
            }],
            weights: vec![0.25],
            biases: vec![0.125],
        };

        assert!(matches!(
            net.to_bytes(),
            Err(NpfError::WeightCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn rejects_too_many_weights() {
        let net = Network {
            header: Header::new("many-weights", [2, 0, 0, 0], [1, 0, 0, 0]),
            layers: vec![Layer::Dense {
                in_features: 2,
                out_features: 1,
            }],
            weights: vec![0.25, -0.5, 0.75],
            biases: vec![0.125],
        };

        assert!(matches!(
            net.to_bytes(),
            Err(NpfError::WeightCountMismatch {
                expected: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn rejects_wrong_bias_count() {
        let net = Network {
            header: Header::new("bad-biases", [2, 0, 0, 0], [1, 0, 0, 0]),
            layers: vec![Layer::Dense {
                in_features: 2,
                out_features: 1,
            }],
            weights: vec![0.25, -0.5],
            biases: vec![],
        };

        assert!(matches!(
            net.to_bytes(),
            Err(NpfError::BiasCountMismatch {
                expected: 1,
                actual: 0
            })
        ));
    }
}
