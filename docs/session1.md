# Session 1 — 2026-03-07

## What we did

### Brainstorming

- Explored use cases for eaclaw + Ea kernels (DB query, vision, network DPI, edge ML)
- Landed on **eaclaw-eye**: WhatsApp-controlled security camera on Raspberry Pi 5
- One binary, zero cloud, L1-resident CNN inference via Ea kernels

### Design decisions

- **Two-stage pipeline:** fused motion detection (full-res) -> fused CNN classify (cropped 64x64)
- **4 classes:** nothing, person, vehicle, animal
- **6KB int8 CNN model** fits in Pi 5 L1d (64KB)
- **Fusion principle:** "If data leaves registers, you probably ended a kernel too early"
  - conv2d+relu+pool = one fused operation
  - GAP+FC+softmax = one fused operation
  - diff+threshold+grid_count = one fused operation (no intermediate diff buffer)
- **Dedup via VectorStore recall** — same system handles "what happened last night?" and alert spam prevention
- **Alert rules configurable via WhatsApp chat** — LLM interprets NL commands
- **Setup:** flash SD, plug in, SSH, run `eaclaw-eye --whatsapp`, scan QR, done

### Hardware

- Pi 5 1GB (549 SEK)
- VESA case (129 SEK)
- 27W USB-C PSU (179 SEK)
- Pi Camera v2 (179 SEK)
- Total: 1036 SEK (~$95)

### Training plan (not implemented yet)

- COCO dataset, auto-crop bounding boxes to 64x64 grayscale
- Train in PyTorch (EyeNet, ~6K params), quantize to int8
- Export flat binary, embed via include_bytes!

## What was built (13 tasks, all complete)

| #   | Component                   | File                    | LOC |
| --- | --------------------------- | ----------------------- | --- |
| 1   | Cargo workspace             | Cargo.toml, CLAUDE.md   | —   |
| 2   | Frame buffer                | vision/frame.rs         | 96  |
| 3   | Camera trait + MockCamera   | vision/capture.rs       | 102 |
| 4   | Fused motion detector       | vision/motion.rs        | 247 |
| 5   | CNN model format (6KB)      | vision/model.rs         | 180 |
| 6   | Fused inference engine      | vision/inference.rs     | 360 |
| 7   | Detection pipeline          | vision/pipeline.rs      | 204 |
| 8   | Alert rules engine          | vision/rules.rs         | 244 |
| 9   | Detection dedup             | vision/dedup.rs         | 142 |
| 10  | Async camera loop           | eye.rs                  | 172 |
| 11  | CLI binary                  | eye-cli/main.rs         | 131 |
| 12  | V4L2 camera (feature-gated) | vision/v4l2.rs          | 68  |
| 13  | Criterion benchmarks        | benches/vision_bench.rs | 56  |

**Totals:** 2014 LOC, 44 tests passing, 5 benchmarks, 11 commits

### Initial benchmarks (dev machine, not Pi)

- Motion detection (640x480): ~1.27ms (~230 MiB/s)
- Crop+scale: ~4.2us
- CNN inference (64x64): ~2.8ms

## Deferred tasks (next sessions)

| #   | Task                   | Notes                                                               |
| --- | ---------------------- | ------------------------------------------------------------------- |
| 14  | WhatsApp image sending | Extend Go bridge + Channel trait for photos                         |
| 15  | Detection history tool | `/detections [N]`, `/stats`                                         |
| 16  | NL alert rule config   | LLM calls `update_rules` tool from chat                             |
| 17  | Ea kernel acceleration | Replace scalar inference with Ea conv2d_3x3, benchmark before/after |
| 18  | Real model training    | COCO dataset, PyTorch, export int8 weights                          |
| 19  | Pi SD card image       | Raspberry Pi OS Lite, auto-start, serial QR                         |

## Key files

- Design doc: `docs/plans/2026-03-07-eaclaw-eye-design.md`
- Implementation plan: `docs/plans/2026-03-07-eaclaw-eye-implementation.md`
- Measurement methodology: `MEASURING.md`
- Engineering rules: `CLAUDE.md`

## Engineering rules (non-negotiable)

1. If it does not compile, it is not a function
2. No file exceeds 500 lines
3. If data leaves registers, you probably ended a kernel too early — FUSION
4. Code with cache in mind — caller-provided buffers, no hot-path allocations
5. Measure everything — same methodology as eaclaw MEASURING.md

MENTAL NOTE: control center in Arch linux hyprspace design.
