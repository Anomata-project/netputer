// SPDX-License-Identifier: Apache-2.0

pub const MAGIC: [u8; 4] = *b"NETP";
pub const VERSION: u32 = 1;
pub const ENDIAN_LE: u32 = 0;
pub const PRECISION_F32: u32 = 32;

// Fixed header bytes excluding the variable-length name field.
pub const HEADER_FIXED_BYTES: usize = 60;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Dense = 0x01,
    Conv2D = 0x02,
    MaxPool2D = 0x03,
    Flatten = 0x04,
    ReLU = 0x10,
    Tanh = 0x11,
    Sigmoid = 0x12,
    Softmax = 0x13,
}

impl LayerType {
    pub fn from_tag(tag: u32) -> Option<Self> {
        Some(match tag {
            0x01 => Self::Dense,
            0x02 => Self::Conv2D,
            0x03 => Self::MaxPool2D,
            0x04 => Self::Flatten,
            0x10 => Self::ReLU,
            0x11 => Self::Tanh,
            0x12 => Self::Sigmoid,
            0x13 => Self::Softmax,
            _ => return None,
        })
    }

    pub fn tag(self) -> u32 {
        self as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    Valid = 0,
    Same = 1,
}

impl PaddingMode {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Valid),
            1 => Some(Self::Same),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    Dense {
        in_features: u32,
        out_features: u32,
    },
    Conv2D {
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        padding_mode: PaddingMode,
    },
    MaxPool2D {
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
    },
    Flatten,
    ReLU,
    Tanh,
    Sigmoid,
    Softmax {
        axis: u32,
    },
}

impl Layer {
    pub fn layer_type(&self) -> LayerType {
        match self {
            Self::Dense { .. } => LayerType::Dense,
            Self::Conv2D { .. } => LayerType::Conv2D,
            Self::MaxPool2D { .. } => LayerType::MaxPool2D,
            Self::Flatten => LayerType::Flatten,
            Self::ReLU => LayerType::ReLU,
            Self::Tanh => LayerType::Tanh,
            Self::Sigmoid => LayerType::Sigmoid,
            Self::Softmax { .. } => LayerType::Softmax,
        }
    }

    pub fn weight_count(&self) -> usize {
        match self {
            Self::Dense {
                in_features,
                out_features,
            } => (*in_features as usize) * (*out_features as usize),
            Self::Conv2D {
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                ..
            } => {
                (*out_channels as usize)
                    * (*in_channels as usize)
                    * (*kernel_h as usize)
                    * (*kernel_w as usize)
            }
            _ => 0,
        }
    }

    pub fn bias_count(&self) -> usize {
        match self {
            Self::Dense { out_features, .. } => *out_features as usize,
            Self::Conv2D { out_channels, .. } => *out_channels as usize,
            _ => 0,
        }
    }

    pub fn param_bytes(&self) -> u32 {
        match self {
            Self::Dense { .. } => 8,
            Self::Conv2D { .. } => 28,
            Self::MaxPool2D { .. } => 16,
            Self::Flatten | Self::ReLU | Self::Tanh | Self::Sigmoid => 0,
            Self::Softmax { .. } => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub endian: u32,
    pub precision: u32,
    pub checksum: u32,
    pub name: String,
    pub input_shape: [u32; 4],
    pub output_shape: [u32; 4],
    pub layer_count: u32,
}

impl Header {
    pub fn new(name: impl Into<String>, input_shape: [u32; 4], output_shape: [u32; 4]) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            endian: ENDIAN_LE,
            precision: PRECISION_F32,
            checksum: 0,
            name: name.into(),
            input_shape,
            output_shape,
            layer_count: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    pub header: Header,
    pub layers: Vec<Layer>,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

impl Network {
    pub fn total_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    pub fn total_weight_count(&self) -> usize {
        self.layers.iter().map(Layer::weight_count).sum()
    }

    pub fn total_bias_count(&self) -> usize {
        self.layers.iter().map(Layer::bias_count).sum()
    }
}
