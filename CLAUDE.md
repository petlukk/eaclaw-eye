# eaclaw-eye — Cache-Resident Security Camera

## Build

cargo build
cargo test
cargo run

## Architecture

Two-stage SIMD pipeline:
1. Motion detection (fused diff+threshold+grid) on full-res frames — rejects ~95%
2. CNN classification (fused conv2d+relu+pool) on cropped 64x64 — person/vehicle/animal/nothing

Depends on eaclaw-core for agent loop, WhatsApp, safety, recall, tools.

## Engineering Rules

1. If it does not compile, it is not a function. No stubs, no TODO, no unimplemented.
2. No file exceeds 500 lines.
3. If data leaves registers, you probably ended a kernel too early. FUSION.
4. Code with cache in mind. Caller-provided buffers, no hot-path allocations. Ea-style.
5. Measure everything. Same methodology as eaclaw MEASURING.md.
