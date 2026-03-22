# Ea Kernel Build Integration

## Summary

Wire the three `.ea` kernel source files (motion, inference, normalize) into a build pipeline that produces a single binary with embedded SIMD kernels. Follows the established eaclaw-core pattern: compile to `.so`, embed via `include_bytes!()`, extract and load at runtime via `libloading`.

## Context

- Ea compiler: `/home/peter/projects/eacompute/target/release/ea` (v1.10.0)
- Supports cross-compilation: `--target-triple=aarch64-unknown-linux-gnu`
- Three kernel sources already written in `kernels/`: `motion.ea`, `inference.ea`, `normalize.ea`
- FFI bridge `accel.rs` exists with safe wrappers but uses direct `extern "C"` linking (won't work)
- Feature-gated integration into `motion.rs` and `inference.rs` already done
- Dev machine: x86_64, deployment target: Raspberry Pi 5 (aarch64) via `scp` + systemd

## Design

### 1. `build.sh` — Kernel Compilation

Compile `.ea` sources to shared libraries for both native (dev/test) and aarch64 (Pi deployment).

```bash
set -euo pipefail
EA="${EA:-/home/peter/projects/eacompute/target/release/ea}"
mkdir -p kernels/prebuilt/x86_64 kernels/prebuilt/aarch64

for f in kernels/*.ea; do
    stem=$(basename "$f" .ea)
    "$EA" "$f" --lib -o "kernels/prebuilt/x86_64/lib${stem}.so"
    "$EA" "$f" --lib --target-triple=aarch64-unknown-linux-gnu \
        -o "kernels/prebuilt/aarch64/lib${stem}.so"
done
```

Output per arch: `lib{motion,inference,normalize}.so`

Pre-built `.so` files are checked into the repo so `cargo build` works without the ea compiler.

### 2. `crates/eye-core/build.rs` — Kernel Embedding

Cargo always runs `build.rs`, so feature-gate internally via `CARGO_FEATURE_EA` env var.

Selects the correct arch subdirectory based on `CARGO_CFG_TARGET_ARCH`. Generates `embedded_kernels.rs`:

```rust
fn main() {
    if std::env::var("CARGO_FEATURE_EA").is_err() {
        return; // no-op when ea feature disabled
    }
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let arch_dir = match arch.as_str() {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        _ => panic!("unsupported target arch: {arch}"),
    };
    // ... find kernels/prebuilt/{arch_dir}/, embed via include_bytes!()
}
```

Output constants: `MOTION`, `INFERENCE`, `NORMALIZE`, `VERSION`.

### 3. Rewrite `accel.rs` — Runtime Loading

Replace `extern "C"` declarations with the eaclaw-core `libloading` pattern:

**Type aliases** (5 functions):

| Function | Library | Signature |
|----------|---------|-----------|
| `motion_fused` | `libmotion.so` | `(*u8, *u8, *mut u32, i32, i32, i32, i32, i32) -> i32` |
| `normalize_u8_f32` | `libnormalize.so` | `(*u8, *mut f32, i32, f32, f32)` |
| `relayout_conv_weights` | `libinference.so` | `(*i8, *mut i8, i32, i32)` |
| `conv3x3_relu_pool` | `libinference.so` | `(*f32, i32, i32, i32, *i8, *i8, i32, *mut f32)` |
| `gap_fc_softmax` | `libinference.so` | `(*f32, i32, i32, i32, *i8, *i8, *mut f32, i32)` |

**`KernelTable`** struct holding function pointers + `Vec<Library>`.

**`OnceLock<KernelTable>`** for thread-safe lazy init.

**`init()`** — extract embedded `.so` to `~/.eaclaw/lib/eye-v{VERSION}/`, load via `libloading`.

**Safe wrappers** — same public API as today (`ea_motion_fused`, `ea_normalize_u8_f32`, etc.).

No changes needed to `vision/mod.rs` (already has `#[cfg(feature = "ea")] pub mod accel`).

### 4. Dependencies

Add to `eye-core/Cargo.toml`:

```toml
libloading = { version = "0.8", optional = true }
home = { version = "0.5", optional = true }

[features]
ea = ["dep:libloading", "dep:home"]
```

### 5. Init Call

`eye-cli/src/main.rs` — add eye kernel init when `ea` feature is active:

```rust
#[cfg(feature = "ea")]
eye_core::vision::accel::init().expect("failed to init eye kernels");
```

## File Changes

| File | Action |
|------|--------|
| `build.sh` | Create — kernel compilation script (both x86_64 + aarch64) |
| `kernels/prebuilt/{x86_64,aarch64}/` | Create dirs, populate with compiled `.so` files |
| `crates/eye-core/build.rs` | Create — embed kernels (gated on `CARGO_FEATURE_EA`) |
| `crates/eye-core/Cargo.toml` | Add `libloading`, `home` as optional deps behind `ea` |
| `crates/eye-core/src/vision/accel.rs` | Rewrite — `extern C` -> `libloading` KernelTable |
| `eye-cli/src/main.rs` | Add eye kernel init call |

## Constraints

- No file exceeds 500 lines
- No hot-path allocations (init extracts once, load once)
- `ea` feature is additive — scalar fallback remains when disabled
- Pre-built `.so` checked in so CI/cargo build works without ea compiler
- Arch-specific `.so` selection via `CARGO_CFG_TARGET_ARCH` in build.rs
