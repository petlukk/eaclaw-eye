# eaclaw-eye Design

WhatsApp-controlled security camera on Raspberry Pi 5. One binary, zero cloud,
entire inference pipeline cache-resident.

## Thesis

For small models on edge devices, a cache-disciplined CPU beats a GPU.
No PCIe transfer, no driver stack, no framework, no VRAM copies — data sits
in L1 next to the ALU. The model is 8KB. The GPU alternative is 200MB.

## Hardware

| Part | Spec | Price |
|------|------|-------|
| Raspberry Pi 5 1GB | Cortex-A76 @ 2.4GHz, 64KB L1d, 512KB L2, 128-bit NEON | 549 SEK |
| VESA case | Wall/monitor mount | 129 SEK |
| 27W USB-C PSU | Official Pi supply | 179 SEK |
| Pi Camera v2 | Sony IMX219, 8MP, 3280x2464, 62deg FOV | 179 SEK |
| **Total** | | **1036 SEK (~$95)** |

## Architecture

SIMD filter + scalar verify, same as eaclaw.

```
Camera (3280x2464)
  |
  v
Stage 1: Fused motion detection (Ea kernel)
  - Fused diff+threshold+grid_count in one pass
  - No intermediate diff buffer — diff value stays in register
  - Outputs bounding boxes of "something moved"
  - Rejects ~95% of frames (nothing happening)
  - L1-resident, nanoseconds
  |
  v
Stage 2: Fused CNN classification (Ea kernels)
  - Crop motion region -> downscale to 64x64
  - 3x fused conv2d+ReLU+pool (conv output stays in register)
  - Fused GAP+FC+softmax (channel averages stay in register)
  - Person / Vehicle / Animal / Nothing
  - int8 weights, ~6KB total, L1-resident
  |
  v
Dedup (VectorStore recall)
  - "Did I already alert about this?"
  - Recall recent detections, suppress if similar
  |
  v
Alert (WhatsApp via eaclaw bridge)
  - Full-res snapshot + timestamp + class
  - "Person detected - 2026-03-07 22:14:03"
```

## Detection Range

Two-stage pipeline extends range beyond the 64x64 constraint:

- Stage 1 (motion): works at any distance the camera resolves
- Stage 2 (classify): runs on the cropped region, not the full frame

Effective range: **20-30m for people, 30-50m for cars** (limited by
camera optics, not model input).

## CNN Model

Purpose-built tiny classifier, int8 quantized, fits in L1d (64KB on Pi 5).

```
Fused Layer 1:  conv2d(3x3x1x8) + ReLU + pool2x2   =    80 bytes
Fused Layer 2:  conv2d(3x3x8x16) + ReLU + pool2x2  =  1.2KB
Fused Layer 3:  conv2d(3x3x16x32) + ReLU + pool2x2 =  4.6KB
Fused Output:   GAP + FC(32x4) + softmax             =    132 bytes
Total weights:                                         ~6KB
```

4 classes: person, vehicle, animal, nothing.
Trained offline, weights ship with the binary.
NEON SDOT instruction: 4 int8 multiply-accumulates per cycle.

**Fusion principle:** If data leaves registers, you probably ended a kernel
too early. conv+relu+pool is one fused operation — conv output stays in a
register, gets ReLU'd, feeds the pool max accumulator, only the pooled
result writes to memory. Same for GAP+FC+softmax: channel averages stay
in registers, immediately multiply into FC accumulators.

## Ea Kernels

| Kernel | Purpose | Source |
|--------|---------|--------|
| sobel | Edge detection for motion stage | eacompute/demo |
| conv2d_3x3 | CNN convolution layers | eacompute/demo |
| pixel_pipeline | Frame preprocessing | eacompute/demo |
| mnist_normalize | Activation normalization | eacompute/demo |
| byte_classifier | Input classification | eaclaw |
| sanitizer | Safety on outbound text | eaclaw |
| leak_scanner | Safety on outbound text | eaclaw |
| search | VectorStore recall for dedup | eaclaw |

## User Interaction (all via WhatsApp)

### Setup
1. Flash SD card with eaclaw-eye image
2. Plug in Pi + camera
3. SSH in, run `eaclaw-eye --whatsapp`
4. Scan QR in terminal
5. Text "hello" — live

### Alert Rules (conversational)
- "only alert me about people"
- "mute until 6pm"
- "ignore cats"
- "be more sensitive"
- "set cooldown to 5 minutes"

LLM (Claude API) interprets these and updates rules.

### Query History
- "anything happen last night?"
- "how many cars today?"
- "show me the last 5 detections"

Uses existing recall system (VectorStore + JSONL persistence).

### Remote Admin (shell tool)
- "restart the camera"
- "disk usage"
- "reboot"
- "tail last 20 alerts"

## Component Reuse from eaclaw

| Component | Reuse |
|-----------|-------|
| WhatsApp bridge (Go, whatsmeow) | As-is |
| Agent loop + tool dispatch | As-is |
| Safety pipeline (sanitizer + leak_scanner) | As-is |
| VectorStore + recall | As-is, pointed at detections |
| JSONL persistence | As-is, detection log |
| Shell tool | As-is |
| LLM integration (Claude API) | As-is, interprets alert rules |
| Tool trait + registry | As-is, add camera/detection tools |
| Kernel FFI + embedding | As-is, add vision kernels |

## New Components

1. **Camera capture** — V4L2 via `v4l` Rust crate, continuous frame grab
2. **Motion detector** — Ea sobel/diff on full-res frames, output bounding boxes
3. **CNN inference** — Ea conv2d + normalize + vec through int8 model
4. **Model weights** — Pre-trained, embedded via include_bytes! (same pattern as kernels)
5. **Detection tool** — "show last N detections", "detection stats"
6. **Alert rules engine** — Parsed from NL by Claude, stored as simple filters
7. **Snapshot capture** — Full-res frame grab + send via WhatsApp

## What Lives Where

```
L1d (64KB):  Ea kernels + model weights (8KB) + working tile + input patch
L1i (64KB):  Ea kernel instructions (~15KB total)
L2 (512KB):  Full-res frame for motion detection, activation buffers
RAM (1GB):   OS, Rust runtime, WhatsApp bridge, JSONL history, snapshots
```

## Engineering Rules

1. **If it does not compile, it is not a function.** No stubs, no `//TODO`,
   no `//HARDCODED`, no `unimplemented!()`. Every function works or does not exist.
2. **No file exceeds 500 lines.** Same as eaclaw.
3. **If data leaves registers, you probably ended a kernel too early. FUSION.**
   Fuse operations that share data in registers. Don't write intermediate results
   to memory between stages. conv2d -> relu -> pool is one kernel, not three.
   Motion diff -> threshold -> bitmask is one kernel, not three.
4. **Code with cache in mind.** Caller-provided buffers, no allocations in hot paths,
   tile-sized working sets, sequential access patterns. Ea-style.
5. **Measure everything.** Same methodology as eaclaw (MEASURING.md):
   - `cargo test` — correctness gate
   - `cargo bench` — Criterion throughput + latency, no regressions
   - `perf stat` — L1i miss rate < 0.01%, IPC > 3.5, branch miss < 0.1%
   - Cycles/byte for inference pipeline
   - Instruction footprint — vision hot path < 8KB
6. **Tests for every module.** Unit tests for all logic. Integration test for
   full pipeline (mock camera -> motion -> classify -> alert).
7. **Benchmarks for hot path.** Criterion benchmarks for:
   - Frame differencing (motion detection)
   - Fused conv2d+relu+pool throughput
   - Full inference pipeline (64x64 -> classification)
   - Crop + downscale

## Not In Scope

- Training (done offline on a real GPU, export int8 weights)
- Multiple cameras (single camera per Pi)
- Web UI (WhatsApp is the only interface)
- Cloud anything
- Night vision (Camera v2 is daylight/IR-cut only)
