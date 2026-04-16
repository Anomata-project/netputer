// SPDX-License-Identifier: Apache-2.0
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NpfError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid magic bytes at offset 0: expected 4E 45 54 50 ('NETP'), found {found:02X?}")]
    BadMagic { found: [u8; 4] },

    #[error("unsupported version at offset 4: got {got}, expected 1")]
    BadVersion { got: u32 },

    #[error("unsupported endian at offset 8: got {got}, expected 0 (little-endian)")]
    BadEndian { got: u32 },

    #[error("unsupported precision at offset 12: got {got}, expected 32 (float32)")]
    BadPrecision { got: u32 },

    #[error("layer_count is 0 at offset {offset}")]
    ZeroLayerCount { offset: u64 },

    #[error("unknown layer_type tag {tag:#010x} at offset {offset}")]
    UnknownLayerType { tag: u32, offset: u64 },

    #[error("CRC32 mismatch: expected {expected:08X}, got {actual:08X}. Weight section may be truncated or corrupted.")]
    CrcMismatch { expected: u32, actual: u32 },

    #[error("input_shape {declared:?} does not match first layer's expected input ({detail})")]
    InputShapeMismatch { declared: [u32; 4], detail: String },

    #[error("output_shape {declared:?} does not match last layer's output ({detail})")]
    OutputShapeMismatch { declared: [u32; 4], detail: String },

    #[error("file ended before all declared sections were complete (truncation at offset {offset})")]
    Truncation { offset: u64 },

    #[error("file has {extra} unexpected bytes after biases section")]
    TrailingBytes { extra: usize },

    #[error("invalid UTF-8 in name field at offset {offset}: {source}")]
    InvalidUtf8Name {
        offset: u64,
        #[source]
        source: std::string::FromUtf8Error,
    },

    #[error("invalid param_bytes for {layer}: expected {expected}, got {got} at offset {offset}")]
    BadParamBytes {
        layer: &'static str,
        expected: u32,
        got: u32,
        offset: u64,
    },

    #[error("invalid padding_mode {got} at offset {offset} (expected 0 = valid or 1 = same)")]
    BadPaddingMode { got: u32, offset: u64 },
}

pub type Result<T> = std::result::Result<T, NpfError>;
