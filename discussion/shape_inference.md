# Shape Inference in NPF v1

## Problem Statement

Rules 8 and 9 in the v1 spec say the declared `input_shape` and `output_shape` must match the dimensions "implied by" the first and last layers. That wording is not precise enough to implement consistently.

The ambiguity appears as soon as a network contains layers that transform spatial dimensions. Example:

- Declared input: `[1, 28, 28, 0]`
- `Conv2D(in=1, out=6, kernel=5x5, stride=1x1, padding=same)`
- `MaxPool2D(kernel=2x2, stride=2x2)`
- `Flatten`
- `Dense(in_features=1176, out_features=120)`

Under standard `same` semantics, the convolution preserves height and width, so the shapes would be:

- Input: `[1, 28, 28, 0]`
- After Conv2D: `[6, 28, 28, 0]`
- After MaxPool2D: `[6, 14, 14, 0]`
- After Flatten: `[1176, 0, 0, 0]`
- After Dense: `[120, 0, 0, 0]`

If the same convolution used `valid` padding instead, the sequence would become:

- After Conv2D: `[6, 24, 24, 0]`
- After MaxPool2D: `[6, 12, 12, 0]`
- After Flatten: `[864, 0, 0, 0]`

That difference changes whether the `Dense(in_features=1176)` layer is valid. So "dimensions implied by" cannot mean only channel count or only the last parameterized layer. It must say whether the parser is expected to infer the shape of every intermediate layer, and if so, exactly how.

## Per-Layer Inference Rules

This section describes the rules a full inference implementation would need.

### Shape Representation

NPF already encodes shapes as four `u32` values with unused dimensions set to `0`. For v1 discussion, a practical convention is:

- 1D tensor: `[features, 0, 0, 0]`
- 3D activation map: `[channels, height, width, 0]`
- Unused dimensions must stay `0`

The spec should say whether zero is only a storage placeholder or may ever represent a real runtime dimension. The current text implies placeholder only.

### Dense

- Expected input: any shape whose product of nonzero dimensions equals `in_features`
- Output shape: `[out_features, 0, 0, 0]`

This is simple, but it deliberately allows multiple equivalent input encodings. `[400, 0, 0, 0]`, `[1, 20, 20, 0]`, and `[10, 10, 4, 0]` all satisfy `in_features = 400`.

### Conv2D

- Expected input shape: `[in_channels, in_h, in_w, 0]`
- Requires `shape[0] == in_channels`
- Output channels: `out_channels`
- Output spatial dimensions depend on padding mode

For `padding=valid`:

- `out_h = floor((in_h - kernel_h) / stride_h) + 1`
- `out_w = floor((in_w - kernel_w) / stride_w) + 1`
- The layer is invalid if `in_h < kernel_h` or `in_w < kernel_w`

For `padding=same`:

- Standard semantic target is `out_h = ceil(in_h / stride_h)`
- `out_w = ceil(in_w / stride_w)`
- This implies the runtime inserts enough zero-padding to preserve size when stride is 1 and otherwise rounds up

Output shape:

- `[out_channels, out_h, out_w, 0]`

The main ambiguity here is whether v1 intends TensorFlow-style `same` semantics exactly, especially for strides greater than 1.

### MaxPool2D

- Expected input shape: `[channels, in_h, in_w, 0]`
- Output channels: unchanged

If pooling uses the common "valid window only" rule:

- `out_h = floor((in_h - kernel_h) / stride_h) + 1`
- `out_w = floor((in_w - kernel_w) / stride_w) + 1`

Output shape:

- `[channels, out_h, out_w, 0]`

The spec currently defines parameters but not rounding behavior for non-divisible dimensions.

### Flatten

- Accepts any input shape
- Output feature count = product of nonzero dimensions
- Output shape: `[flattened, 0, 0, 0]`

This is the bridge that makes internal consistency checks useful. A wrong Conv2D or MaxPool2D shape will surface here as the wrong flattened feature count.

### ReLU

- Output shape = input shape

### Tanh

- Output shape = input shape

### Sigmoid

- Output shape = input shape

### Softmax

- Output shape = input shape

For Softmax, shape inference is trivial, but axis validation is not. The spec does not currently define which axes are legal for encoded 1D vs 3D shapes or how zero placeholder dimensions affect axis numbering.

## Design Options

### Option A: Minimal

Validate only what the current code effectively validates:

- `Dense`: compare product of nonzero dims with `in_features` or `out_features`
- `Conv2D`: compare only channel count against declared input/output shape
- Ignore intermediate shape transitions
- Do not reason about MaxPool2D, Flatten, or activation layers

Pros:

- Very small parser
- Low risk of introducing bugs in v1 tooling
- Keeps runtime and validator behavior simple

Cons:

- Files with internal dimension mismatches can pass validation
- Rules 8 and 9 remain only partially enforced
- Human readers may assume stronger guarantees than the format actually provides

### Option B: Full Inference

Compute the output shape of every layer in order:

- Check that the declared `input_shape` exactly matches the first layer's expected input
- Verify every layer accepts the previous layer's output
- Check that the declared `output_shape` exactly matches the actual final output
- Reject internal mismatches such as `Flatten -> Dense` count errors

Pros:

- Strongest correctness guarantees
- Makes Rules 8 and 9 operationally precise
- Catches malformed files before load time

Cons:

- More parser code and more spec text
- Forces decisions on currently ambiguous semantics (`same`, pooling rounding, Softmax axis rules)
- Raises the bar for alternate implementations that want strict compatibility

### Option C: Middle Ground

Split checks into mandatory core validation and optional deep validation.

Possible structure:

- Mandatory:
  - Exact checks for Dense, Flatten, and obvious Conv2D channel mismatches
  - Exact declared `input_shape` and `output_shape` matching when the shapes can be inferred without ambiguity
  - Rejection of internal `Flatten -> Dense` count mismatches
- Optional:
  - Full spatial inference through Conv2D and MaxPool2D
  - Strict Softmax axis validation

Pros:

- Better protection than Option A
- Lower implementation burden than Option B
- Lets the validator be stricter than a tiny OS loader

Cons:

- Two compliance tiers can confuse users
- "Optional" checks are easy to ignore in practice
- The spec must say very clearly which failures are format-invalid versus validator-only warnings

## Implementation Cost

### Option A

Low. The parser already does most of it. Cost is mainly spec clarification and perhaps minor cleanup.

### Option B

Moderate. The parser needs:

- A shape type and helper functions
- Conv2D and MaxPool2D spatial arithmetic
- Internal layer-by-layer validation
- New error variants for internal mismatches
- Additional tests for edge cases

The code is still small in absolute terms, but the semantic surface area grows noticeably.

### Option C

Moderate, but with policy complexity instead of pure algorithmic complexity. The hard part is deciding where the mandatory/optional line sits and keeping tooling behavior understandable.

## Runtime Cost

All options are cheap relative to model execution. The real question is where the work belongs.

- OS loader: should probably perform at least the mandatory safety checks for any file it accepts
- Standalone validator: can afford full inference and better diagnostics because it runs once on a development machine

If the project wants a minimal loader, that argues toward either Option C or a spec that separates "must reject" from "recommended validator checks".

## Spec Implications

Whichever option is chosen, the spec needs sharper language in at least these areas:

- Define the canonical meaning of `[d0, d1, d2, d3]` for v1 tensors
- Define Conv2D `same` semantics precisely, especially for stride > 1
- Define MaxPool2D output rounding and invalid-size behavior
- Define whether Flatten always produces `[N, 0, 0, 0]`
- Define Softmax axis validity against NPF's shape encoding
- Clarify whether Rules 8 and 9 concern only declared endpoint shapes or also internal shape consistency

If Option B is chosen, the Validation Rules section should gain explicit internal mismatch language rather than relying only on "first layer" and "last layer" wording.

If Option C is chosen, the spec likely needs two categories:

- Format validity rules
- Recommended validator checks

## Open Questions

- Should v1 have one canonical tensor layout only: `[channels, height, width, 0]` for spatial data and `[features, 0, 0, 0]` for flat data?
- Is `Dense` allowed to consume any shape with the right element count, or only `[features, 0, 0, 0]` after an explicit Flatten?
- Is TensorFlow-style `same` the intended Conv2D behavior, including ceil-based output sizing for stride > 1?
- What is the intended MaxPool2D rule when the window does not divide the input evenly?
- Should Softmax axis be validated in v1, and against which logical rank?
- Must the OS loader enforce the same shape rules as the offline validator, or is stricter validation allowed only in the validator?

## Main Tension

The central design choice is correctness versus simplicity: stronger shape guarantees require precise semantics for every layer type, while a lighter v1 parser can stay small only by accepting that some malformed networks will still look valid.
