# Netputer

A tensor-native operating system for small neural networks on cheap hardware.

Netputer is not a Linux distribution. It is not a Python library. It is not a training framework. It is a bare-metal OS where neural activations are the native execution model — not a library bolted onto a von Neumann machine.

You bring a trained network. Netputer runs it. That is the whole idea.

---

## What this is not

- **Not a training environment** — train on your development machine, deploy here
- **Not for large models** — if your network has more than ~1M parameters, this is the wrong tool
- **Not a Linux distro** — no POSIX, no filesystem, no process isolation
- **Not general purpose** — you cannot browse the web or compile code on a Netputer

---

## What it is

A Raspberry Pi (or any 64-bit machine) running a custom OS that:

- Loads networks from `.npf` files — a flat binary format containing architecture + weights, nothing else
- Connects networks directly to physical inputs (GPIO, camera, microphone) and network sources (HTTP, MQTT)
- Runs 100+ tiny networks concurrently with an activation-based scheduler
- Displays live gradient flows and activation state on screen
- Boots to inference in under 100ms

---

## The .npf format

Every network is a single `.npf` file (Netputer Package Format). Self-describing flat binary. No runtime dependencies. No external schema. Typically 10KB–500KB.

Any tool that produces a spec-compliant `.npf` file can target Netputer — a training framework exporter, an LLM agent, a hand-written binary. The spec is the contract.

→ [NPF Format Specification v1.0](./spec/npf-spec-v1.0.md)

---

## Networks that will run on day one

| Network | Year | Params | Task |
|---------|------|--------|------|
| Perceptron | 1958 | ~1K | Binary classification |
| ADALINE | 1960 | ~100 | Adaptive filtering |
| LeNet-1 | 1989 | 10K | Digit recognition |
| LeNet-5 | 1998 | 60K | MNIST |
| Elman net | 1990 | 5–50K | Sequence prediction |
| Jordan net | 1990 | 5–50K | Sequence prediction |
| TDNN | 1989 | 50K | Phoneme recognition |
| Neurogammon | 1991 | 1K | Backgammon |

We are not competing with GPT. We are resurrecting the 1990s vision of cheap, ubiquitous, embedded neural networks.

---

## Status

**Pre-prototype. Spec phase.**

The `.npf` format spec is written and stable. No OS code exists yet. Milestone 1 is bare-metal boot on Raspberry Pi 5 with a working tensor memory allocator.

---

## Roadmap

| # | Milestone |
|---|-----------|
| 1 | Bare metal boot on RPi 5 + tensor memory allocator |
| 2 | Load and run LeNet-5 |
| 3 | Input/output ports (GPIO, serial, HTTP) |
| 4 | Activation scheduler — multiple concurrent networks |
| 5 | Weight store + `.npf` file loader |
| 6 | Dashboard UI |
| 7 | Offline conversion tool (runs on dev machine) |

---

## Contributing

This project needs people who know things that are not commonly combined:

- **Bare-metal ARM / Raspberry Pi** — bootloader, memory layout, NEON intrinsics
- **Neural network internals** — forward pass implementations, especially Conv2D and recurrent layers
- **Classic network history** — if you have run a LeNet or Elman net and understand the original papers, that knowledge is directly useful here
- **Spec review** — read the `.npf` spec and tell us what is wrong or missing

Open an issue or start a discussion. There is no code to PR against yet — that is intentional.

---

## Further reading

- [White Paper](./docs/netputer-whitepaper.md) — full concept, history, and use cases
- [NPF Format Specification v1.0](./spec/npf-spec-v1.0.md) — the file format in full detail
