<!-- SPDX-License-Identifier: Apache-2.0 -->
# npf-toolkit

A Rust workspace for working with `.npf` files — the Netputer Package Format.

- **`npf`** — library for reading and writing `.npf` files with full validation.
- **`npf-validate`** — CLI tool that validates a `.npf` file against the spec.
- **`npf-gen`** — CLI tool that generates example `.npf` files (tiny, mlp, lenet5) for testing.

The authoritative format definition lives in [`spec/npf-spec-v1.3.md`](../../spec/npf-spec-v1.3.md).

## Build

```
cargo build --workspace
cargo test --workspace
```

## Usage

```
cargo run -p npf-gen -- lenet5
cargo run -p npf-validate -- lenet5.npf
```
