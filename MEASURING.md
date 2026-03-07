# eaclaw-eye Measurement Methodology

Same methodology as eaclaw. Every hot-path change gets measured.

---

## 1. Tests (correctness gate)

```bash
cargo test
```

---

## 2. Criterion Benchmarks (throughput + latency)

```bash
cargo bench -p eye-core
```

Benchmarks:
- motion_detection: fused diff+threshold+grid at 640x480, 1280x720, 1920x1080
- crop_and_scale: 640x480 region to 64x64
- cnn_inference: fused forward pass on 64x64 frame

---

## 3. perf stat: Cache + Branch + IPC

```bash
cargo build --release -p eye-core --example perf_vision

perf stat -e instructions,cycles,L1-icache-load-misses,L1-dcache-load-misses,branches,branch-misses \
  target/release/examples/perf_vision 100000
```

Targets:
- L1i miss rate < 0.01%
- L1d miss rate < 0.1% (model weights stay in cache)
- IPC > 3.5
- Branch miss rate < 0.1%

---

## 4. Instruction Footprint

```bash
nm -S --size-sort target/release/examples/perf_vision | grep -E 'conv|motion|classify|pool|fused'
```

Target: vision hot path < 8KB instructions.

---

## Measurement Checklist

After every hot-path change:

1. [ ] cargo test — all pass
2. [ ] cargo bench — no regressions
3. [ ] perf stat — L1i miss < 0.01%, IPC > 3.5
4. [ ] Code size — hot path < 8KB
5. [ ] Update BENCHMARKS.md with new results
