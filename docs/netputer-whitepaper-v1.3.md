<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Netputer: A Tensor-Native Operating System for Small Neural Networks
**White Paper v1.3**

---

## Executive Summary

The Netputer is a purpose-built operating system for running many small neural networks on cheap, ubiquitous hardware — Raspberry Pi, thin clients, edge devices. It treats neural activations as the native execution model, not as a library running on top of a general-purpose OS.

**Core insight:** We are entering a third wave of neural network adoption. The first wave (1950s–60s) proved the math. The second wave (1980s–90s) proved the algorithms. The third wave (2020s) makes them cheap and everywhere. Netputer is the OS for this third wave.

**Target audience:** Edge AI developers who want fast, local, reliable inference without OS overhead. Educators. Neural network historians. Hobbyists with clusters of Raspberry Pis.

---

## A Historical Note: Three Waves of Neural Networks

### First Wave: The Foundations (1950s–1960s)

- **McCulloch-Pitts neuron (1943)** — The formal neuron. Proved that networks of simple binary units could compute any logical function.
- **Perceptron (1958, Frank Rosenblatt)** — The first trainable neural network. Ran on custom IBM hardware (the Mark I Perceptron). Could recognize simple shapes.
- **ADALINE (1960, Bernard Widrow)** — Adaptive linear neuron. Used in real applications: adaptive filters, echo cancellation.

What they got right: neural networks are fundamentally different from sequential computers. They deserve their own hardware and software stack.

What they got wrong: single-layer networks are too limited. The field collapsed after Minsky and Papert's critique (1969).

**Netputer connection:** The Mark I Perceptron was a netputer — dedicated hardware, no filesystem, direct sensor inputs, visual activation display. We are reclaiming that vision with modern hardware.

### Second Wave: The Algorithms (1980s–1990s)

- **Backpropagation (1986, Rumelhart, Hinton, Williams)** — Made multi-layer networks trainable.
- **Convolutional networks (1989, LeCun et al.)** — LeNet-5 recognized handwritten digits on 8-bit microprocessors with no GPU.
- **Recurrent networks (1990s)** — Elman networks, Jordan networks, and LSTMs (1997) for sequence learning.
- **Commercial applications** — Check reading (banking), speech recognition, fraud detection.

A LeNet-5 from 1989 still classifies MNIST digits with ~99% accuracy. A small Elman network from 1990 still learns simple sequences. These networks are tiny by today's standards — 10K to 100K parameters — which means they fit perfectly on a Raspberry Pi netputer.

**Netputer connection:** The Netputer is designed to run exactly these kinds of networks — small, fast, single-purpose. We are not competing with large-scale deep learning. We are resurrecting the 1990s vision of ubiquitous, cheap, embedded neural networks.

### Third Wave: 2012–Present

AlexNet, TensorFlow, PyTorch, Transformers, and large language models proved that scale works. But most applications do not need scale. Edge computing makes small, fast, local inference valuable again, and modern tooling makes exporting small networks easier than ever.

**Netputer connection:** The Netputer is the deployment target for this third-wave edge renaissance.

---

## The Problem Netputer Solves

Current edge AI deployment looks like this:

```
Neural net → train → export → deploy via Docker → run under Linux
→ compete with hundreds of irrelevant system processes
→ waste most of your cycles on OS overhead
```

You are not using a computer to run neural nets. You are using a general-purpose computer that tolerates neural nets as an afterthought.

**Specific pains:**
- Tensor memory fragmented by OS virtual memory
- Context switching destroys activation state
- No native support for running 100+ tiny nets concurrently
- Debugging requires symbolic tools that don't understand gradients
- Deployment requires significant DevOps infrastructure

Netputer eliminates all of this.

---

## The Solution: Netputer

A Netputer machine (minimum viable: Raspberry Pi 5 + custom OS) provides:

### 1. Self-describing networks
Every network is a single `.npf` file — the Netputer Package Format — containing the complete architecture and trained weights. Load it. Run it. No separate model definition. No runtime dependencies. (See Appendix B for format details.)

### 2. Native input/output ports
Networks connect directly to physical sensors (GPIO, camera, microphone), network sources (HTTP, MQTT, TCP/UDP), and other networks. No glue code. No API layer. The scheduler routes tensor data from ports to networks to ports.

### 3. Tensor-native memory
Physical memory is partitioned into tensor regions, not pages. Allocation is O(1) and cache-aware. No fragmentation. No virtual memory overhead.

### 4. Activation-scheduled execution
The scheduler runs activation flows, not threads. Each network is a graph. The scheduler walks the graph, executing ready nodes when input tensors arrive.

### 5. Weight-first persistence
The entire system state is a collection of `.npf` files. Boot-to-inference in under 100ms.

### 6. Gradient-native introspection
Every tensor knows its gradient. Every activation can be traced. Debugging is visual and interactive — you can watch gradients flow through your network in real time.

---

## What You Bring to Netputer

You bring a single `.npf` file — a self-contained binary containing both architecture and weights. You can generate it by:

- Training on any platform and exporting via our offline conversion tool
- Generating it directly against the open spec with any tool of your choice
- Writing it by hand (possible but not recommended)

Netputer does not consume ONNX, TensorFlow, or PyTorch models. You convert them offline using a separate tool that runs on your normal development machine. The Netputer itself never sees those formats. The `.npf` spec is open — any tool that produces a valid file can target Netputer.

---

## What You See When You Connect a Screen

> **Note:** The screens in this section are aspirational mockups illustrating the intended user experience. No working implementation exists yet. They are included to make the design vision concrete for contributors and reviewers.

**Boot screen (first 2 seconds)**

```
NETPUTER v0.1 — Raspberry Pi 5
─────────────────────────────────
Weight store loaded: 47 networks
Memory: 7.2GB free / 8GB total
Input ports: GPIO, CAM0, ETH0, USB, MQTT
─────────────────────────────────
Ready.
```

**Main dashboard**

```
┌─────────────────────────────────────────────┐
│ NETPUTER — Activation Monitor               │
├─────────────────────────────────────────────┤
│                                             │
│ Active networks: ████░░░░░░ 4/47            │
│ Total FPS: 342 (inferences per second)      │
│ Avg latency: 2.3ms per net                  │
│                                             │
│ ┌─────────────────────────────────────┐     │
│ │ net_001: LeNet-5 (1998)             │     │
│ │ Input: 28x28 (waiting)              │     │
│ │ Last: "7" (94% confidence)          │     │
│ │ ─────────────────────────────────── │     │
│ │ net_012: Elman_recur (1990)         │     │
│ │ Input: sequence (active)            │     │
│ │ Last prediction: 0.87               │     │
│ │ ─────────────────────────────────── │     │
│ │ net_033: temp_predictor             │     │
│ │ Input: GPIO12 (22.3°C)              │     │
│ │ Output: 23.1°C (1 min forecast)     │     │
│ └─────────────────────────────────────┘     │
│                                             │
│ [Tab] cycle views  [Enter] inspect net      │
│ [Ctrl+W] weight editor  [Ctrl+V] visualize  │
└─────────────────────────────────────────────┘
```

**Activation visualization (Ctrl+V)**

```
net_012: Elman_recur — ACTIVATION FLOW
─────────────────────────────────────────
Input: sequence[0] = 0.42
↓
Hidden layer (16 units)
→ ████░░░░░░░░░░░░ 4/16 active
↓
Context layer (feedback)
→ ████████░░░░░░░░ 8/16 active
↓
Output (1 unit)
→ 0.67
─────────────────────────────────────────
[Space] pause  [G] show gradients  [W] edit weights
```

---

## Hardware: Raspberry Pi First, Then Beyond

We test on:
- **Raspberry Pi 5** — cheap, available, runs bare metal
- **Used office computers** — any 64-bit machine from the last 10 years works

No special hardware. No expensive accelerators. The whole point is to prove what small neural nets can do on hardware people already have or can acquire for nearly nothing.

---

## Use Cases

### 1. Historical Preservation & Education
Run actual networks from the 1990s on modern cheap hardware. LeNet-5. Elman networks. Jordan networks. The TDNN from 1989. See them work. Understand the roots of deep learning.

### 2. Sensor Fusion (Smart Home / Agriculture)
20 sensors, each with a tiny predictor. Netputer runs all 20 networks simultaneously at 10Hz. Total cost: one Raspberry Pi.

### 3. Educational Cluster
30 students, each running a different tiny network. Netputer time-slices activation flows, giving each student a virtual netputer. No cloud. No GPUs.

### 4. Edge AI Microservices
Replace service containers with networks. Each service is a tiny neural net. Netputer routes inputs to nets, outputs to other nets. Activation flow is the message bus.

---

## What Netputer Is Not

- **Not a Linux distribution** — no POSIX, no traditional filesystem, no process isolation
- **Not a training environment** — you train on your development machine and deploy here
- **Not for large models** — if your network has more than ~1M parameters, Netputer is the wrong tool
- **Not general purpose** — you cannot browse the web, edit documents, or compile code on a Netputer

---

## Development Roadmap

Milestones are sequenced by dependency, not by calendar.

| # | Milestone |
|---|-----------|
| 1 | Bare metal boot on RPi 5 + tensor memory allocator |
| 2 | Load and run a simple network (LeNet-5 as benchmark) |
| 3 | Input/output ports (GPIO, serial, HTTP) |
| 4 | Activation scheduler — multiple concurrent networks |
| 5 | Weight store + `.npf` file loader |
| 6 | Dashboard UI |
| 7 | Offline conversion tool (runs on dev machine) |

---

## Call to Action

This white paper is an invitation.

**For neural network historians:** Help us port the classic networks from the 1950s–1990s. Let's make them run again.

**For embedded systems developers:** Help us optimize the tensor memory allocator, the NEON vector kernels, the scheduler.

**For educators:** Help us design the curriculum. A Netputer in every classroom, running the networks that started it all.

**For you (the reader):** Build the prototype. One Raspberry Pi. One working network. Then show someone else.

The third wave is here. The tools are cheap. The historical knowledge is waiting.

Let's build the netputer.

---

## Appendix A: Classic Networks That Will Run Day One

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

Each of these will run on a Raspberry Pi netputer. Many will run hundreds of copies simultaneously.

---

## Appendix B: The .npf Format (Sketch)

The Netputer Package Format is an open flat binary format. Any tool that produces a valid `.npf` file can target Netputer — the format is the contract, not any particular toolchain.

### File layout

```
┌────────────────────────────────────────────┐
│ HEADER                                     │
│   Magic:      "NETP" (4 bytes, ASCII)      │
│   Version:    uint32 (currently 1)         │
│   Endian:     uint32 (0 = little-endian)   │
│   Precision:  uint32 (32 = float32)        │
│   Checksum:   uint32 (CRC32 of weight sec) │
│   Name len:   uint32                       │
│   Name:       N bytes, UTF-8               │
│                (no terminator, no padding) │
│   Input dims: 4 × uint32                   │
│   Output dims:4 × uint32                   │
│   Layer count:uint32                       │
├────────────────────────────────────────────┤
│ ARCHITECTURE                               │
│   Layer 1:  type tag (uint32)              │
│             params (varies by type)        │
│   Layer 2:  ...                            │
│   ...                                      │
├────────────────────────────────────────────┤
│ WEIGHTS                                    │
│   Flat float32 array, layer order          │
├────────────────────────────────────────────┤
│ BIASES                                     │
│   Flat float32 array, layer order          │
└────────────────────────────────────────────┘
```

### Layer types (v1)

| Tag | Type | Key params |
|-----|------|------------|
| 0x01 | Dense | in_features, out_features |
| 0x02 | Conv2D | in_ch, out_ch, kernel_h, kernel_w, stride_h, stride_w, padding |
| 0x03 | MaxPool2D | kernel_h, kernel_w, stride_h, stride_w |
| 0x04 | Flatten | — |
| 0x10 | ReLU | — |
| 0x11 | Tanh | — |
| 0x12 | Sigmoid | — |
| 0x13 | Softmax | axis |

Unknown type tags cause the runtime to reject the file with a clear error. This is intentional — silent wrong behaviour is worse than a hard stop.

### Hex example: minimal header

```
4E 45 54 50         ← magic "NETP"
01 00 00 00         ← version 1
00 00 00 00         ← little-endian
20 00 00 00         ← float32
A3 4F 2B 11         ← CRC32 checksum
05 00 00 00         ← name length 5
6D 79 6E 65 74      ← "mynet" (5 bytes, no terminator, no padding)
...
```

### What is not in v1

Recurrent layers, BatchNorm, Conv1D, and attention mechanisms are out of scope for v1. The version field in the header ensures future layer types can be added without breaking existing files. A v1 runtime encountering an unknown layer tag rejects the file and reports the unknown tag — it does not attempt to run a partially understood network.

The full specification is maintained as a separate document: **NPF Format Specification v1.3**.

---

## License

This white paper is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt it for any purpose, including commercially, provided you give appropriate credit to the Netputer project.
