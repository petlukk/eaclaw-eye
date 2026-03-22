# eaclaw-eye

Cache-resident security camera for Raspberry Pi 5. Two-stage SIMD pipeline: fused motion detection rejects ~95% of frames, then a 6KB CNN classifies the rest as person/vehicle/animal/nothing. Alerts via WhatsApp.

Part of the [eaclaw](https://github.com/petlukk/eaclaw) project.

## Architecture

```
Camera (V4L2 or mock)
    |
[Stage 1: Fused Motion Detection]     ~122µs @ 640x480
  abs-diff + threshold + grid count    single pass, diff stays in register
  output: bounding boxes               rejects ~95% of frames
    |
[Stage 2: Fused CNN Classification]    ~2.4ms @ 64x64
  crop region → 64x64
  3× conv3x3+ReLU+MaxPool (fused)     one write per pool window
  GAP+FC+softmax (fused)              channel avg stays in register
  output: class + confidence
    |
[Dedup → Rules → Alert]
  cosine similarity dedup
  class/confidence/cooldown filters
  WhatsApp via eaclaw agent
```

The CNN model is 6KB int8 — fits entirely in the Pi 5's L1d cache (64KB).

Hot paths are compiled as [Ea](https://github.com/petlukk/eacompute) SIMD kernels (ARM NEON on Pi, SSE2/AVX on x86). Scalar Rust fallback when the `ea` feature is disabled.

## Build

```bash
cargo build                    # scalar fallback
cargo build --features ea      # with Ea SIMD kernels
cargo build --features v4l2    # with Pi Camera v2 support
cargo test
cargo bench -p eye-core
```

### Ea Kernels

Pre-built kernels for x86_64 and aarch64 are in `kernels/prebuilt/`. To rebuild from source:

```bash
./build.sh                     # requires ea compiler
```

## Run

```bash
# Mock camera (testing)
cargo run -- --mock

# Real camera on Pi
cargo run --features v4l2,ea

# With WhatsApp alerts
cargo run --features v4l2,ea -- --whatsapp
```

Requires `ANTHROPIC_API_KEY` in environment.

## Deploy (Raspberry Pi 5)

```bash
# Cross-compile
cargo build --release --features v4l2,ea --target aarch64-unknown-linux-gnu

# Copy to Pi
scp target/aarch64-unknown-linux-gnu/release/eaclaw-eye pi@<host>:

# On the Pi: run as systemd service
sudo cp eaclaw-eye.service /etc/systemd/system/
sudo systemctl enable --now eaclaw-eye
```

## Performance

Benchmarked on x86_64 (scalar → Ea SIMD):

| Stage | Scalar | Ea SIMD | Speedup |
|-------|--------|---------|---------|
| Motion 640x480 | 1.27ms | 122µs | 10.4x |
| Motion 1920x1080 | 10.9ms | 1.84ms | 5.9x |
| CNN inference 64x64 | 2.8ms | 2.43ms | 1.15x |
| Crop+scale | 4.8µs | 4.8µs | — |

See [MEASURING.md](MEASURING.md) for methodology.

## Project Structure

```
crates/eye-core/src/
  vision/
    motion.rs      Fused diff+threshold+grid motion detection
    inference.rs   Fused conv+relu+pool CNN engine
    model.rs       6KB int8 model format and loading
    pipeline.rs    Two-stage detection pipeline
    rules.rs       Alert filtering (class/confidence/cooldown)
    dedup.rs       Cosine similarity dedup via VectorStore
    frame.rs       Frame buffer with crop+scale
    capture.rs     Camera trait + MockCamera
    v4l2.rs        V4L2 capture (feature-gated)
    accel.rs       Ea kernel FFI bridge (feature-gated)
  eye.rs           Async camera loop + event channel
eye-cli/src/
  main.rs          CLI binary with agent integration
kernels/
  motion.ea        SIMD motion detection kernel
  inference.ea     SIMD conv+pool+GAP+FC+softmax kernels
  normalize.ea     SIMD u8→f32 normalization
training/
  train.py         CNN training script (COCO dataset)
```

## License

MIT
