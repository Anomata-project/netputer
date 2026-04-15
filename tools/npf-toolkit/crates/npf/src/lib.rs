// SPDX-License-Identifier: Apache-2.0
//! Reader, writer, and validator for the Netputer Package Format (.npf).
//!
//! The authoritative format definition lives in `spec/npf-spec-v1.3.md` at the
//! repository root.

pub mod error;
pub mod parse;
pub mod types;

pub use error::{NpfError, Result};
pub use types::{
    Header, Layer, LayerType, Network, PaddingMode, ENDIAN_LE, HEADER_FIXED_BYTES, MAGIC,
    PRECISION_F32, VERSION,
};
