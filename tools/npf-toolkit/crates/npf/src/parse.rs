// SPDX-License-Identifier: Apache-2.0
use crate::error::{NpfError, Result};
use crate::types::{
    Header, Layer, LayerType, Network, PaddingMode, ENDIAN_LE, MAGIC, PRECISION_F32, VERSION,
};

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> u64 {
        self.pos as u64
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(NpfError::Truncation {
                offset: self.pos as u64,
            });
        }
        Ok(())
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let s = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_u32_array4(&mut self) -> Result<[u32; 4]> {
        Ok([
            self.read_u32()?,
            self.read_u32()?,
            self.read_u32()?,
            self.read_u32()?,
        ])
    }

    fn read_f32(&mut self) -> Result<f32> {
        let s = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}

fn read_layer(c: &mut Cursor) -> Result<Layer> {
    let record_offset = c.pos();
    let tag = c.read_u32()?;
    let param_bytes = c.read_u32()?;

    let ty = LayerType::from_tag(tag).ok_or(NpfError::UnknownLayerType {
        tag,
        offset: record_offset,
    })?;

    let expected_param_bytes = match ty {
        LayerType::Dense => 8,
        LayerType::Conv2D => 28,
        LayerType::MaxPool2D => 16,
        LayerType::Flatten => 0,
        LayerType::ReLU => 0,
        LayerType::Tanh => 0,
        LayerType::Sigmoid => 0,
        LayerType::Softmax => 4,
    };
    if param_bytes != expected_param_bytes {
        return Err(NpfError::BadParamBytes {
            layer: match ty {
                LayerType::Dense => "Dense",
                LayerType::Conv2D => "Conv2D",
                LayerType::MaxPool2D => "MaxPool2D",
                LayerType::Flatten => "Flatten",
                LayerType::ReLU => "ReLU",
                LayerType::Tanh => "Tanh",
                LayerType::Sigmoid => "Sigmoid",
                LayerType::Softmax => "Softmax",
            },
            expected: expected_param_bytes,
            got: param_bytes,
            offset: record_offset + 4,
        });
    }

    let layer = match ty {
        LayerType::Dense => Layer::Dense {
            in_features: c.read_u32()?,
            out_features: c.read_u32()?,
        },
        LayerType::Conv2D => {
            let in_channels = c.read_u32()?;
            let out_channels = c.read_u32()?;
            let kernel_h = c.read_u32()?;
            let kernel_w = c.read_u32()?;
            let stride_h = c.read_u32()?;
            let stride_w = c.read_u32()?;
            let pad_offset = c.pos();
            let pad_raw = c.read_u32()?;
            let padding_mode = PaddingMode::from_u32(pad_raw).ok_or(NpfError::BadPaddingMode {
                got: pad_raw,
                offset: pad_offset,
            })?;
            Layer::Conv2D {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_mode,
            }
        }
        LayerType::MaxPool2D => Layer::MaxPool2D {
            kernel_h: c.read_u32()?,
            kernel_w: c.read_u32()?,
            stride_h: c.read_u32()?,
            stride_w: c.read_u32()?,
        },
        LayerType::Flatten => Layer::Flatten,
        LayerType::ReLU => Layer::ReLU,
        LayerType::Tanh => Layer::Tanh,
        LayerType::Sigmoid => Layer::Sigmoid,
        LayerType::Softmax => Layer::Softmax {
            axis: c.read_u32()?,
        },
    };

    Ok(layer)
}

impl Network {
    pub fn parse(bytes: &[u8]) -> Result<Network> {
        let mut c = Cursor::new(bytes);

        // ---- Header ----
        let magic_slice = c.read_bytes(4)?;
        let mut magic = [0u8; 4];
        magic.copy_from_slice(magic_slice);
        if magic != MAGIC {
            return Err(NpfError::BadMagic { found: magic });
        }

        let version = c.read_u32()?;
        if version != VERSION {
            return Err(NpfError::BadVersion { got: version });
        }

        let endian = c.read_u32()?;
        if endian != ENDIAN_LE {
            return Err(NpfError::BadEndian { got: endian });
        }

        let precision = c.read_u32()?;
        if precision != PRECISION_F32 {
            return Err(NpfError::BadPrecision { got: precision });
        }

        let checksum = c.read_u32()?;

        let name_len = c.read_u32()? as usize;
        let name_offset = c.pos();
        let name_bytes = c.read_bytes(name_len)?.to_vec();
        let name = String::from_utf8(name_bytes).map_err(|source| NpfError::InvalidUtf8Name {
            offset: name_offset,
            source,
        })?;

        let input_shape = c.read_u32_array4()?;
        let output_shape = c.read_u32_array4()?;

        let layer_count_offset = c.pos();
        let layer_count = c.read_u32()?;
        if layer_count == 0 {
            return Err(NpfError::ZeroLayerCount {
                offset: layer_count_offset,
            });
        }

        // ---- Architecture ----
        let mut layers: Vec<Layer> = Vec::with_capacity(layer_count as usize);
        for _ in 0..layer_count {
            layers.push(read_layer(&mut c)?);
        }

        // ---- Weights ----
        let total_weight_count: usize = layers.iter().map(Layer::weight_count).sum();
        let total_weight_bytes = total_weight_count * 4;
        c.need(total_weight_bytes)?;
        let weight_bytes_start = c.pos() as usize;
        let weight_raw = &bytes[weight_bytes_start..weight_bytes_start + total_weight_bytes];

        let actual_crc = crc32fast::hash(weight_raw);
        if actual_crc != checksum {
            return Err(NpfError::CrcMismatch {
                expected: checksum,
                actual: actual_crc,
            });
        }

        let mut weights = Vec::with_capacity(total_weight_count);
        for _ in 0..total_weight_count {
            weights.push(c.read_f32()?);
        }

        // ---- Biases ----
        let total_bias_count: usize = layers.iter().map(Layer::bias_count).sum();
        c.need(total_bias_count * 4)?;
        let mut biases = Vec::with_capacity(total_bias_count);
        for _ in 0..total_bias_count {
            biases.push(c.read_f32()?);
        }
        if c.remaining() != 0 {
            return Err(NpfError::TrailingBytes {
                extra: c.remaining(),
            });
        }

        let header = Header {
            magic,
            version,
            endian,
            precision,
            checksum,
            name,
            input_shape,
            output_shape,
            layer_count,
        };

        Ok(Network {
            header,
            layers,
            weights,
            biases,
        })
    }

    pub fn from_reader<R: std::io::Read>(mut r: R) -> Result<Network> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        Self::parse(&buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal valid Dense(2→1) + ReLU network byte-by-byte.
    // Used as a fixture that individual tests then mutate to trigger each rule.
    fn build_tiny_bytes() -> Vec<u8> {
        let weights: [f32; 2] = [0.5, -0.25];
        let biases: [f32; 1] = [0.125];
        let mut weight_bytes = Vec::new();
        for w in weights {
            weight_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let crc = crc32fast::hash(&weight_bytes);

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC); // magic
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&0u32.to_le_bytes()); // endian LE
        out.extend_from_slice(&32u32.to_le_bytes()); // precision
        out.extend_from_slice(&crc.to_le_bytes()); // checksum
        out.extend_from_slice(&4u32.to_le_bytes()); // name_len
        out.extend_from_slice(b"test"); // name
                                        // input_shape [2,0,0,0]
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        // output_shape [1,0,0,0]
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        // layer_count = 2
        out.extend_from_slice(&2u32.to_le_bytes());

        // Dense layer
        out.extend_from_slice(&0x01u32.to_le_bytes()); // type
        out.extend_from_slice(&8u32.to_le_bytes()); // param_bytes
        out.extend_from_slice(&2u32.to_le_bytes()); // in_features
        out.extend_from_slice(&1u32.to_le_bytes()); // out_features

        // ReLU layer
        out.extend_from_slice(&0x10u32.to_le_bytes()); // type
        out.extend_from_slice(&0u32.to_le_bytes()); // param_bytes

        // Weights
        out.extend_from_slice(&weight_bytes);
        // Biases
        for b in biases {
            out.extend_from_slice(&b.to_le_bytes());
        }

        out
    }

    #[test]
    fn parses_minimal_valid_file() {
        let bytes = build_tiny_bytes();
        let net = Network::parse(&bytes).expect("should parse");
        assert_eq!(net.header.name, "test");
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.weights.len(), 2);
        assert_eq!(net.biases.len(), 1);
    }

    // Rule 1: magic
    #[test]
    fn rule_1_bad_magic() {
        let mut bytes = build_tiny_bytes();
        bytes[0] = b'X';
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::BadMagic { .. })
        ));
    }

    // Rule 2: version
    #[test]
    fn rule_2_bad_version() {
        let mut bytes = build_tiny_bytes();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::BadVersion { got: 2 })
        ));
    }

    // Rule 3: endian
    #[test]
    fn rule_3_bad_endian() {
        let mut bytes = build_tiny_bytes();
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::BadEndian { got: 1 })
        ));
    }

    // Rule 4: precision
    #[test]
    fn rule_4_bad_precision() {
        let mut bytes = build_tiny_bytes();
        bytes[12..16].copy_from_slice(&16u32.to_le_bytes());
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::BadPrecision { got: 16 })
        ));
    }

    // Rule 5: zero layer_count
    #[test]
    fn rule_5_zero_layer_count() {
        let mut bytes = build_tiny_bytes();
        // layer_count is at offset 24 + name_len(4) + 16 + 16 = 60, 4 bytes
        let off = 60;
        bytes[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::ZeroLayerCount { .. })
        ));
    }

    // Rule 6: unknown layer_type
    #[test]
    fn rule_6_unknown_layer_type() {
        let mut bytes = build_tiny_bytes();
        // First layer tag at offset 64 (right after layer_count)
        let off = 64;
        bytes[off..off + 4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::UnknownLayerType {
                tag: 0xDEADBEEF,
                ..
            })
        ));
    }

    // Rule 7: CRC mismatch
    #[test]
    fn rule_7_crc_mismatch() {
        let mut bytes = build_tiny_bytes();
        // Flip a byte in the weights section. Weights start at:
        // 60 + name_len(4) header + 16 Dense + 8 ReLU = 88
        let w_off = 88;
        bytes[w_off] ^= 0xFF;
        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::CrcMismatch { .. })
        ));
    }

    // Rule 8: truncation
    #[test]
    fn rule_8_truncation() {
        let bytes = build_tiny_bytes();
        let short = &bytes[..bytes.len() - 2];
        assert!(matches!(
            Network::parse(short),
            Err(NpfError::Truncation { .. })
        ));
    }

    #[test]
    fn rejects_trailing_bytes_after_biases() {
        let mut bytes = build_tiny_bytes();
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);

        assert!(matches!(
            Network::parse(&bytes),
            Err(NpfError::TrailingBytes { extra: 3 })
        ));
    }
}
