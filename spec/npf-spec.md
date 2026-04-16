<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# NPF Format Specification
**Netputer Package Format — v1.4**

---

## Overview

The Netputer Package Format (`.npf`) is the native binary format for neural networks on the Netputer OS. A single `.npf` file is fully self-describing: it contains the complete network architecture, all trained weights and biases, and enough metadata to validate and execute the network without any external dependencies.

The format is open. Any tool that produces a spec-compliant `.npf` file can target Netputer — a training framework exporter, a code-generation tool, a hand-written binary, or a dedicated conversion utility. The spec is the contract.

---

## Design Principles

- **Container with semantic neutrality.** The format specifies a binary container — magic, version, layer records, weights, biases, checksum. It does not specify the semantics of layer operators or the shape relationships between layers. A valid `.npf` file is a well-formed container. What the tensors mean is a contract between the file's author and the runtime that loads it. Optional helper tools may impose stricter checks; the format itself does not.
- **Flat binary.** No compression, no nested containers, no external schema. The file is readable with a hex editor.
- **Self-validating.** A CRC32 checksum covers the weight section. The runtime rejects corrupt or truncated files with a clear error before executing anything.
- **Forward-compatible.** The version field allows future spec revisions. Unknown layer type tags cause an explicit rejection — the runtime never silently ignores unknown sections.
- **Little-endian throughout.** Netputer targets ARM hardware. All multi-byte integers and floats are stored little-endian. The endian field in the header is reserved for documentation and validation; a big-endian file is invalid in v1.
- **float32 throughout.** All weights and biases are 32-bit IEEE 754 floats. The precision field reserves space for future float16 support; a file declaring any precision other than 32 is invalid in v1.

---

## File Structure

A valid `.npf` file consists of four sections in strict order:

```
[ HEADER ] [ ARCHITECTURE ] [ WEIGHTS ] [ BIASES ]
```

There is no padding between sections. Offsets are computed from declared sizes.

---

## Section 1: Header

Total minimum size: 60 bytes + name length.

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | ASCII | magic | Always `4E 45 54 50` ("NETP"). Reject if absent. |
| 4 | 4 | uint32 | version | Format version. Currently `1`. |
| 8 | 4 | uint32 | endian | `0` = little-endian. Any other value is invalid in v1. |
| 12 | 4 | uint32 | precision | `32` = float32. Any other value is invalid in v1. |
| 16 | 4 | uint32 | checksum | CRC32 of the WEIGHTS section (bytes only, not BIASES). |
| 20 | 4 | uint32 | name_len | Byte length of the name field. May be 0. |
| 24 | name_len | UTF-8 | name | Human-readable network name. No null terminator. No padding. Empty string (name_len = 0) is valid. |
| 24 + name_len | 16 | 4 × uint32 | input_shape | Up to 4 dimensions. Unused dimensions set to 0. Example: a 28×28 grayscale image = `[1, 28, 28, 0]`. |
| 40 + name_len | 16 | 4 × uint32 | output_shape | Same encoding as input_shape. |
| 56 + name_len | 4 | uint32 | layer_count | Number of layer records in the ARCHITECTURE section. |

### Notes

- The runtime must read `name_len` before attempting to read the name field.
- The name field contains exactly `name_len` bytes of UTF-8 data. There is no null terminator and no padding. The next field (`input_shape`) begins immediately after the last byte of the name.
- `input_shape` and `output_shape` are informational metadata. Runtimes and helper tools may use them for display or additional checks, but shape agreement is not part of container validation.
- `layer_count` of 0 is invalid.

---

## Section 2: Architecture

The architecture section is a sequence of `layer_count` layer records. Each record begins with a fixed preamble followed by type-specific parameters.

### Layer record preamble (8 bytes)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | uint32 | layer_type | Type tag (see table below). |
| 4 | 4 | uint32 | param_bytes | Byte length of the type-specific parameter block that follows. |

Immediately following the preamble: `param_bytes` bytes of type-specific parameters.

An unknown `layer_type` tag causes the runtime to reject the file immediately. It does not skip the layer and continue.

### Layer type tags (v1)

#### 0x00000001 — Dense

Fully connected layer.

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 4 | uint32 | in_features |
| 4 | 4 | uint32 | out_features |

`param_bytes` = 8.

Weight count: `in_features × out_features`.
Bias count: `out_features`.

#### 0x00000002 — Conv2D

2D convolutional layer.

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 4 | uint32 | in_channels |
| 4 | 4 | uint32 | out_channels |
| 8 | 4 | uint32 | kernel_h |
| 12 | 4 | uint32 | kernel_w |
| 16 | 4 | uint32 | stride_h |
| 20 | 4 | uint32 | stride_w |
| 24 | 4 | uint32 | padding_mode | `0` = valid (no padding), `1` = same |

`param_bytes` = 28.

Weight count: `out_channels × in_channels × kernel_h × kernel_w`.
Bias count: `out_channels`.
Weight layout: `[out_ch][in_ch][kernel_h][kernel_w]`, row-major.

#### 0x00000003 — MaxPool2D

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 4 | uint32 | kernel_h |
| 4 | 4 | uint32 | kernel_w |
| 8 | 4 | uint32 | stride_h |
| 12 | 4 | uint32 | stride_w |

`param_bytes` = 16.

No weights. No biases.

#### 0x00000004 — Flatten

Reshapes input tensor to 1D for the following Dense layer.

`param_bytes` = 0. No parameters. No weights. No biases.

#### 0x00000010 — ReLU

Element-wise activation: `max(0, x)`.

`param_bytes` = 0. No weights. No biases.

#### 0x00000011 — Tanh

Element-wise activation: `tanh(x)`.

`param_bytes` = 0. No weights. No biases.

#### 0x00000012 — Sigmoid

Element-wise activation: `1 / (1 + exp(-x))`.

`param_bytes` = 0. No weights. No biases.

#### 0x00000013 — Softmax

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 4 | uint32 | axis | Axis along which to apply softmax. |

`param_bytes` = 4. No weights. No biases.

---

## Section 3: Weights

A flat, contiguous array of float32 values. All weights from all layers are concatenated in layer order. Within each layer, weight layout follows the convention defined for that layer type (see Architecture section above).

Activation layers (ReLU, Tanh, Sigmoid, Softmax), MaxPool2D, and Flatten contribute zero bytes to this section.

Total byte count: `(sum of all layer weight counts) × 4`.

The CRC32 checksum in the header covers exactly this section — the raw bytes from the first weight float to the last, inclusive.

---

## Section 4: Biases

A flat, contiguous array of float32 values. All biases from all layers are concatenated in layer order.

Only Dense and Conv2D layers contribute to this section. All other layer types contribute zero bytes.

Total byte count: `(sum of all layer bias counts) × 4`.

There is no checksum on the bias section in v1.

---

## Validation Rules

A runtime must reject a file — with a descriptive error, not silent wrong output — if any of the following are true:

1. Magic bytes are not `4E 45 54 50`
2. Version is not `1`
3. Endian is not `0`
4. Precision is not `32`
5. `layer_count` is `0`
6. Any layer record contains an unknown `layer_type` tag
7. CRC32 of the weight section does not match the header checksum
8. File ends before all declared sections are complete (truncation)
9. File contains trailing bytes after the biases section

---

## What The Format Does Not Validate

The format does not validate:

- that layers' shapes are mutually consistent (for example, that a Dense layer's `in_features` matches the feature count produced by a preceding Flatten)
- that declared `input_shape` or `output_shape` match the dimensions a particular runtime would compute for the network
- that layer parameters are meaningful under any specific semantic convention (for example, TensorFlow-style versus PyTorch-style Conv2D padding)

These concerns belong to the runtime or to optional helper tools. A file that passes container validation may still fail at load time on a particular runtime due to semantic mismatches. This is intentional. Imposing semantic rules in the format would constrain what kinds of networks can be expressed in NPF, which is the opposite of what the format is for.

---

## Hex Reference: Minimal Valid File

A single Dense(2→1) network with ReLU, name "test":

```
4E 45 54 50   magic "NETP"
01 00 00 00   version 1
00 00 00 00   little-endian
20 00 00 00   float32
XX XX XX XX   CRC32 (computed over weight section)
04 00 00 00   name_len = 4
74 65 73 74   "test" (exactly 4 bytes, no terminator, no padding)
02 00 00 00   input_shape [2, 0, 0, 0]
00 00 00 00
00 00 00 00
00 00 00 00
01 00 00 00   output_shape [1, 0, 0, 0]
00 00 00 00
00 00 00 00
00 00 00 00
02 00 00 00   layer_count = 2

── ARCHITECTURE ──────────────────────────
01 00 00 00   layer_type = Dense
08 00 00 00   param_bytes = 8
02 00 00 00   in_features = 2
01 00 00 00   out_features = 1

10 00 00 00   layer_type = ReLU
00 00 00 00   param_bytes = 0

── WEIGHTS ───────────────────────────────
[2 × float32: weights for Dense layer]

── BIASES ────────────────────────────────
[1 × float32: bias for Dense layer]
```

---

## What Is Not in v1

The following are explicitly deferred to v2 or later:

- Recurrent layer types (Elman, Jordan, LSTM, GRU)
- BatchNorm and LayerNorm
- Conv1D and Conv3D
- Attention and transformer blocks
- float16 precision
- Sparse weight storage
- Multiple input or output ports within a single file

The version field in the header is the extension mechanism. A v2 file loaded by a v1 runtime will fail at rule 2 above with a clear version mismatch error.

---

## Version History

| Version | Changes |
|---------|---------|
| v1.4 | Removed shape-matching validation rules (former rules 8 and 9). Introduced "container with semantic neutrality" as an explicit design principle. Binary format unchanged — header version field remains 1. |
| v1.3 | Header size prose corrected to 60 bytes + name length. |

---

## Reference Validator

A standalone reference validator (runs on any platform, not part of the Netputer OS) is the recommended way to verify a `.npf` file before loading it onto a device. The validator checks all rules in the Validation Rules section above and reports the first failing rule with byte offset.

Validator usage:

```
npf-validate mynetwork.npf
→ OK: mynetwork.npf (LeNet-5, 60450 params, 242KB)

npf-validate corrupt.npf
→ ERROR: CRC32 mismatch. Expected A3 4F 2B 11, got 00 00 00 00.
         Weight section may be truncated or corrupted.
```

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | — | Initial release. Dense, Conv2D, MaxPool2D, Flatten, ReLU, Tanh, Sigmoid, Softmax. |
| 1.3 | — | Documentation aligned across spec, whitepaper, and README. Clarified that the name field has no null terminator and no padding. Binary format unchanged (header `version` field remains `1`). |

---

## License

This specification is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt it for any purpose, including commercially, provided you give appropriate credit to the Netputer project.
