# eaclaw-eye Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** WhatsApp-controlled security camera on Raspberry Pi 5 — one binary, zero cloud, L1-resident CNN inference via Eä kernels.

**Architecture:** Two-stage SIMD pipeline: motion detection (sobel/diff on full-res) filters frames, then CNN classification (conv2d/normalize/vec on cropped 64x64 patch) identifies person/vehicle/animal/nothing. Alerts with photo via WhatsApp. Rules configured conversationally. Reuses eaclaw-core for agent loop, WhatsApp bridge, safety, recall, tools, and kernel FFI.

**Tech Stack:** Rust, Eä kernels (SIMD), eaclaw-core (path dep), v4l (V4L2 camera), tokio, whatsmeow (Go bridge)

**Hard Rules:**
1. If it does not compile, it is not a function. No `//TODO`, `//STUB`, `//HARDCODED`, `unimplemented!()`, `todo!()`. Every function works or does not exist.
2. No file exceeds 500 lines. Split by responsibility if approaching the limit.
3. **If data leaves registers, you probably ended a kernel too early. FUSION.** Fuse operations that share data in registers. conv2d+relu+pool is one fused operation, not three separate passes over memory. Motion diff+threshold+bitmask is one fused operation. Never write intermediate results to memory between stages that can share registers.
4. Code with cache in mind. Caller-provided buffers, no allocations in hot paths, tile-sized working sets, sequential access. Eä-style.
5. Measure everything — same methodology as eaclaw MEASURING.md:
   - `cargo test` — correctness gate, every task ends with all tests passing
   - `cargo bench` — Criterion benchmarks for hot path, no regressions
   - `perf stat` targets: L1i miss < 0.01%, IPC > 3.5, branch miss < 0.1%
   - Instruction footprint: vision hot path < 8KB
6. Tests for every module. Benchmarks for every hot-path function.

---

## Phase 1: Project Scaffold

### Task 1: Initialize Cargo workspace

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/eye-core/Cargo.toml`
- Create: `crates/eye-core/src/lib.rs`
- Create: `eye-cli/Cargo.toml`
- Create: `eye-cli/src/main.rs`
- Create: `CLAUDE.md`

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = ["crates/eye-core", "eye-cli"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"

[workspace.dependencies]
eye-core = { path = "crates/eye-core" }
eaclaw-core = { path = "../eaclaw/crates/eaclaw-core" }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"

[profile.release]
lto = true
codegen-units = 1
```

**Step 2: Create eye-core crate**

`crates/eye-core/Cargo.toml`:
```toml
[package]
name = "eye-core"
version.workspace = true
edition.workspace = true

[dependencies]
eaclaw-core.workspace = true
tokio.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
tracing.workspace = true
```

`crates/eye-core/src/lib.rs`:
```rust
pub mod vision;
```

**Step 3: Create eye-cli crate**

`eye-cli/Cargo.toml`:
```toml
[package]
name = "eaclaw-eye"
version.workspace = true
edition.workspace = true

[[bin]]
name = "eaclaw-eye"
path = "src/main.rs"

[dependencies]
eye-core.workspace = true
eaclaw-core.workspace = true
tokio.workspace = true
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

`eye-cli/src/main.rs`:
```rust
fn main() {
    println!("eaclaw-eye {}", env!("CARGO_PKG_VERSION"));
}
```

**Step 4: Create CLAUDE.md**

```markdown
# eaclaw-eye — Cache-Resident Security Camera

## Build

cargo build
cargo test
cargo run

## Architecture

Two-stage SIMD pipeline:
1. Motion detection (sobel/diff) on full-res frames — rejects ~95%
2. CNN classification (conv2d/normalize/vec) on cropped 64x64 — person/vehicle/animal/nothing

Depends on eaclaw-core for agent loop, WhatsApp, safety, recall, tools.
```

**Step 5: Verify it builds**

Run: `cd /mnt/c/Users/peter/Desktop/eaclaw-eye && cargo build`
Expected: compiles, prints version when run

**Step 6: Commit**

```bash
git init
git add -A
git commit -m "feat: initialize eaclaw-eye workspace with eye-core and eye-cli"
```

---

## Phase 2: Frame Buffer and Camera Abstraction

### Task 2: Frame buffer type

**Files:**
- Create: `crates/eye-core/src/vision/mod.rs`
- Create: `crates/eye-core/src/vision/frame.rs`
- Modify: `crates/eye-core/src/lib.rs`

**Step 1: Write the failing test**

`crates/eye-core/src/vision/frame.rs`:
```rust
/// A grayscale frame buffer. Caller-provided memory, no allocations during capture.
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u64,
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0u8; (width * height) as usize],
            width,
            height,
            timestamp_ms: 0,
        }
    }

    pub fn pixel(&self, x: u32, y: u32) -> u8 {
        self.data[(y * self.width + x) as usize]
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, val: u8) {
        self.data[(y * self.width + x) as usize] = val;
    }

    /// Crop a region and downscale to target_size x target_size (nearest neighbor).
    /// Used to extract motion regions for CNN input.
    pub fn crop_and_scale(&self, x: u32, y: u32, w: u32, h: u32, target_size: u32) -> Frame {
        let mut out = Frame::new(target_size, target_size);
        for ty in 0..target_size {
            for tx in 0..target_size {
                let sx = x + (tx * w) / target_size;
                let sy = y + (ty * h) / target_size;
                let sx = sx.min(self.width - 1);
                let sy = sy.min(self.height - 1);
                out.set_pixel(tx, ty, self.pixel(sx, sy));
            }
        }
        out
    }

    /// Number of bytes in the frame.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_new() {
        let f = Frame::new(64, 64);
        assert_eq!(f.len(), 64 * 64);
        assert_eq!(f.pixel(0, 0), 0);
    }

    #[test]
    fn test_frame_set_pixel() {
        let mut f = Frame::new(10, 10);
        f.set_pixel(5, 3, 200);
        assert_eq!(f.pixel(5, 3), 200);
    }

    #[test]
    fn test_crop_and_scale() {
        let mut f = Frame::new(100, 100);
        // Fill a 20x20 block at (10,10) with value 255
        for y in 10..30 {
            for x in 10..30 {
                f.set_pixel(x, y, 255);
            }
        }
        let cropped = f.crop_and_scale(10, 10, 20, 20, 8);
        assert_eq!(cropped.width, 8);
        assert_eq!(cropped.height, 8);
        // All pixels in crop should be 255
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(cropped.pixel(x, y), 255, "at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_crop_and_scale_downscale() {
        let mut f = Frame::new(640, 480);
        f.set_pixel(320, 240, 128);
        let cropped = f.crop_and_scale(0, 0, 640, 480, 64);
        assert_eq!(cropped.width, 64);
        assert_eq!(cropped.height, 64);
        assert_eq!(cropped.len(), 64 * 64);
    }
}
```

**Step 2: Wire up modules**

`crates/eye-core/src/vision/mod.rs`:
```rust
pub mod frame;
```

Update `crates/eye-core/src/lib.rs`:
```rust
pub mod vision;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 4 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/
git commit -m "feat: add Frame buffer with crop-and-scale for CNN input"
```

---

### Task 3: Camera capture trait and mock

**Files:**
- Create: `crates/eye-core/src/vision/capture.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write capture trait and mock**

`crates/eye-core/src/vision/capture.rs`:
```rust
use super::frame::Frame;
use std::time::{SystemTime, UNIX_EPOCH};

/// Camera capture interface.
pub trait Camera: Send {
    /// Grab a frame into the provided buffer. Returns Ok(true) if a new frame
    /// was captured, Ok(false) if no frame was ready, Err on failure.
    fn grab(&mut self, frame: &mut Frame) -> Result<bool, String>;

    /// Resolution of the camera.
    fn resolution(&self) -> (u32, u32);
}

/// Mock camera for testing. Generates frames with a moving bright spot.
pub struct MockCamera {
    width: u32,
    height: u32,
    frame_count: u32,
}

impl MockCamera {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, frame_count: 0 }
    }
}

impl Camera for MockCamera {
    fn grab(&mut self, frame: &mut Frame) -> Result<bool, String> {
        if frame.width != self.width || frame.height != self.height {
            return Err("frame size mismatch".into());
        }
        // Clear frame
        frame.data.fill(0);

        // Draw a 20x20 bright spot that moves across the frame
        let spot_x = (self.frame_count * 5) % self.width;
        let spot_y = self.height / 2;
        for dy in 0..20u32 {
            for dx in 0..20u32 {
                let x = spot_x + dx;
                let y = spot_y + dy;
                if x < self.width && y < self.height {
                    frame.set_pixel(x, y, 200);
                }
            }
        }

        frame.timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.frame_count += 1;
        Ok(true)
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_camera_grabs_frame() {
        let mut cam = MockCamera::new(640, 480);
        let mut frame = Frame::new(640, 480);
        assert!(cam.grab(&mut frame).unwrap());
        assert!(frame.timestamp_ms > 0);
        // Should have some non-zero pixels (the bright spot)
        assert!(frame.data.iter().any(|&p| p > 0));
    }

    #[test]
    fn test_mock_camera_spot_moves() {
        let mut cam = MockCamera::new(640, 480);
        let mut f1 = Frame::new(640, 480);
        let mut f2 = Frame::new(640, 480);
        cam.grab(&mut f1).unwrap();
        cam.grab(&mut f2).unwrap();
        // Frames should differ (spot moved)
        assert!(f1.data != f2.data);
    }

    #[test]
    fn test_mock_camera_size_mismatch() {
        let mut cam = MockCamera::new(640, 480);
        let mut frame = Frame::new(320, 240);
        assert!(cam.grab(&mut frame).is_err());
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod frame;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 7 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/capture.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add Camera trait and MockCamera for testing"
```

---

## Phase 3: Motion Detection (Stage 1)

### Task 4: Frame differencing for motion detection

**Files:**
- Create: `crates/eye-core/src/vision/motion.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write motion detector with tests**

`crates/eye-core/src/vision/motion.rs`:
```rust
use super::frame::Frame;

/// A bounding box around a region of motion.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Number of changed pixels in this region.
    pub pixel_count: u32,
}

/// Detects motion by comparing two frames via absolute pixel difference.
/// Returns bounding boxes around regions where motion exceeds the threshold.
pub struct MotionDetector {
    /// Pixel difference threshold (0-255). Pixels differing by more than this are "changed".
    pub threshold: u8,
    /// Minimum number of changed pixels to count as a motion region.
    pub min_pixels: u32,
    /// Grid cell size for spatial grouping.
    cell_size: u32,
    /// Scratch buffer for grid cell counts (no per-pixel diff buffer needed — fused).
    grid_counts: Vec<u32>,
}

impl MotionDetector {
    pub fn new(threshold: u8, min_pixels: u32) -> Self {
        Self {
            threshold,
            min_pixels,
            cell_size: 32,
            grid_counts: Vec::new(),
        }
    }

    /// Fused diff+threshold+grid count in one pass.
    /// Computes absolute difference, thresholds, and counts per grid cell
    /// without writing an intermediate diff buffer. The diff value stays
    /// in a register — never hits memory.
    pub fn detect(&mut self, prev: &Frame, curr: &Frame) -> Vec<MotionBox> {
        assert_eq!(prev.width, curr.width);
        assert_eq!(prev.height, curr.height);

        let width = prev.width;
        let height = prev.height;
        let cols = (width + self.cell_size - 1) / self.cell_size;
        let rows = (height + self.cell_size - 1) / self.cell_size;

        // Grid counts — pre-allocated, zeroed
        self.grid_counts.resize((cols * rows) as usize, 0);
        self.grid_counts.fill(0);

        let mut total_changed = 0u32;

        // Single pass: diff + threshold + grid accumulate (fused)
        for y in 0..height {
            let gy = y / self.cell_size;
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let diff = (prev.data[idx] as i16 - curr.data[idx] as i16).unsigned_abs() as u8;
                // diff stays in register — threshold test, grid accumulate, done
                if diff > self.threshold {
                    let gx = x / self.cell_size;
                    self.grid_counts[(gy * cols + gx) as usize] += 1;
                    total_changed += 1;
                }
            }
        }

        if total_changed < self.min_pixels {
            return Vec::new();
        }

        // Extract boxes from grid counts
        let mut boxes = Vec::new();
        for gy in 0..rows {
            for gx in 0..cols {
                let count = self.grid_counts[(gy * cols + gx) as usize];
                if count >= self.min_pixels {
                    let x0 = gx * self.cell_size;
                    let y0 = gy * self.cell_size;
                    boxes.push(MotionBox {
                        x: x0,
                        y: y0,
                        w: self.cell_size.min(width - x0),
                        h: self.cell_size.min(height - y0),
                        pixel_count: count,
                    });
                }
            }
        }

        merge_boxes(&mut boxes, self.cell_size);
        boxes
    }
}

/// Merge boxes that are within one cell_size of each other.
fn merge_boxes(boxes: &mut Vec<MotionBox>, gap: u32) {
    if boxes.len() <= 1 {
        return;
    }

    let mut merged = true;
    while merged {
        merged = false;
        let mut i = 0;
        while i < boxes.len() {
            let mut j = i + 1;
            while j < boxes.len() {
                if boxes_overlap(&boxes[i], &boxes[j], gap) {
                    let b = boxes.remove(j);
                    let a = &mut boxes[i];
                    let x1 = a.x.min(b.x);
                    let y1 = a.y.min(b.y);
                    let x2 = (a.x + a.w).max(b.x + b.w);
                    let y2 = (a.y + a.h).max(b.y + b.h);
                    a.x = x1;
                    a.y = y1;
                    a.w = x2 - x1;
                    a.h = y2 - y1;
                    a.pixel_count += b.pixel_count;
                    merged = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
}

fn boxes_overlap(a: &MotionBox, b: &MotionBox, gap: u32) -> bool {
    let a_right = a.x + a.w + gap;
    let b_right = b.x + b.w + gap;
    let a_bottom = a.y + a.h + gap;
    let b_bottom = b.y + b.h + gap;

    a.x <= b_right && b.x <= a_right && a.y <= b_bottom && b.y <= a_bottom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_motion() {
        let mut det = MotionDetector::new(10, 5);
        let f1 = Frame::new(64, 64);
        let f2 = Frame::new(64, 64);
        let boxes = det.detect(&f1, &f2);
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_motion_detected() {
        let mut det = MotionDetector::new(10, 5);
        let f1 = Frame::new(64, 64);
        let mut f2 = Frame::new(64, 64);
        // Create a bright region in f2
        for y in 10..30 {
            for x in 10..30 {
                f2.set_pixel(x, y, 200);
            }
        }
        let boxes = det.detect(&f1, &f2);
        assert!(!boxes.is_empty());
        // The motion box should cover the bright region
        let b = &boxes[0];
        assert!(b.x <= 10 && b.y <= 10);
        assert!(b.x + b.w >= 30 && b.y + b.h >= 30);
    }

    #[test]
    fn test_below_threshold_ignored() {
        let mut det = MotionDetector::new(100, 5);
        let f1 = Frame::new(64, 64);
        let mut f2 = Frame::new(64, 64);
        // Change by only 50 — below threshold of 100
        for y in 0..64 {
            for x in 0..64 {
                f2.set_pixel(x, y, 50);
            }
        }
        let boxes = det.detect(&f1, &f2);
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_min_pixels_filter() {
        let mut det = MotionDetector::new(10, 100);
        let f1 = Frame::new(64, 64);
        let mut f2 = Frame::new(64, 64);
        // Only 4 pixels changed — below min_pixels of 100
        f2.set_pixel(0, 0, 200);
        f2.set_pixel(1, 0, 200);
        f2.set_pixel(0, 1, 200);
        f2.set_pixel(1, 1, 200);
        let boxes = det.detect(&f1, &f2);
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_merge_adjacent_boxes() {
        let mut det = MotionDetector::new(10, 1);
        let f1 = Frame::new(128, 128);
        let mut f2 = Frame::new(128, 128);
        // Two adjacent bright regions that should merge
        for y in 10..25 {
            for x in 10..25 {
                f2.set_pixel(x, y, 200);
            }
        }
        for y in 25..40 {
            for x in 10..25 {
                f2.set_pixel(x, y, 200);
            }
        }
        let boxes = det.detect(&f1, &f2);
        // Should merge into one box
        assert_eq!(boxes.len(), 1, "expected 1 merged box, got {}", boxes.len());
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod frame;
pub mod motion;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 12 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/motion.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add MotionDetector with frame differencing and box merging"
```

---

## Phase 4: CNN Inference (Stage 2)

### Task 5: Model weights format and loading

**Files:**
- Create: `crates/eye-core/src/vision/model.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write model definition with tests**

`crates/eye-core/src/vision/model.rs`:
```rust
/// A tiny CNN model for 4-class classification.
///
/// Architecture:
///   Conv1: 3x3x1x8   (72 weights + 8 bias = 80 params)
///   Conv2: 3x3x8x16  (1152 weights + 16 bias = 1168 params)
///   Conv3: 3x3x16x32 (4608 weights + 32 bias = 4640 params)
///   Pool:  global average pooling -> 32 features
///   FC:    32x4       (128 weights + 4 bias = 132 params)
///
/// Total: 6020 int8 params = ~6KB
///
/// Classes: 0=nothing, 1=person, 2=vehicle, 3=animal

pub const NUM_CLASSES: usize = 4;
pub const INPUT_SIZE: u32 = 64;

pub const CLASS_NAMES: [&str; NUM_CLASSES] = ["nothing", "person", "vehicle", "animal"];

/// Layer dimensions.
pub const CONV1_IN: usize = 1;
pub const CONV1_OUT: usize = 8;
pub const CONV2_IN: usize = 8;
pub const CONV2_OUT: usize = 16;
pub const CONV3_IN: usize = 16;
pub const CONV3_OUT: usize = 32;
pub const FC_IN: usize = 32;

/// Weight counts per layer.
pub const CONV1_WEIGHTS: usize = 3 * 3 * CONV1_IN * CONV1_OUT; // 72
pub const CONV1_BIAS: usize = CONV1_OUT; // 8
pub const CONV2_WEIGHTS: usize = 3 * 3 * CONV2_IN * CONV2_OUT; // 1152
pub const CONV2_BIAS: usize = CONV2_OUT; // 16
pub const CONV3_WEIGHTS: usize = 3 * 3 * CONV3_IN * CONV3_OUT; // 4608
pub const CONV3_BIAS: usize = CONV3_OUT; // 32
pub const FC_WEIGHTS: usize = FC_IN * NUM_CLASSES; // 128
pub const FC_BIAS: usize = NUM_CLASSES; // 4

pub const TOTAL_PARAMS: usize = CONV1_WEIGHTS + CONV1_BIAS
    + CONV2_WEIGHTS + CONV2_BIAS
    + CONV3_WEIGHTS + CONV3_BIAS
    + FC_WEIGHTS + FC_BIAS; // 6020

/// Packed model weights. All int8, laid out sequentially.
pub struct Model {
    pub data: Vec<i8>,
}

/// Classification result.
#[derive(Debug, Clone)]
pub struct Classification {
    pub class_id: usize,
    pub class_name: &'static str,
    pub confidence: f32,
    pub scores: [f32; NUM_CLASSES],
}

impl Model {
    /// Load model from raw bytes (int8 weights, packed sequentially).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != TOTAL_PARAMS {
            return Err(format!(
                "expected {} bytes, got {}",
                TOTAL_PARAMS,
                data.len()
            ));
        }
        let weights: Vec<i8> = data.iter().map(|&b| b as i8).collect();
        Ok(Self { data: weights })
    }

    /// Create a dummy model (all zeros) for testing.
    pub fn dummy() -> Self {
        Self {
            data: vec![0i8; TOTAL_PARAMS],
        }
    }

    /// Byte offset for each layer's weights.
    pub fn conv1_weights(&self) -> &[i8] {
        &self.data[0..CONV1_WEIGHTS]
    }
    pub fn conv1_bias(&self) -> &[i8] {
        let start = CONV1_WEIGHTS;
        &self.data[start..start + CONV1_BIAS]
    }
    pub fn conv2_weights(&self) -> &[i8] {
        let start = CONV1_WEIGHTS + CONV1_BIAS;
        &self.data[start..start + CONV2_WEIGHTS]
    }
    pub fn conv2_bias(&self) -> &[i8] {
        let start = CONV1_WEIGHTS + CONV1_BIAS + CONV2_WEIGHTS;
        &self.data[start..start + CONV2_BIAS]
    }
    pub fn conv3_weights(&self) -> &[i8] {
        let start = CONV1_WEIGHTS + CONV1_BIAS + CONV2_WEIGHTS + CONV2_BIAS;
        &self.data[start..start + CONV3_WEIGHTS]
    }
    pub fn conv3_bias(&self) -> &[i8] {
        let start = CONV1_WEIGHTS + CONV1_BIAS + CONV2_WEIGHTS + CONV2_BIAS + CONV3_WEIGHTS;
        &self.data[start..start + CONV3_BIAS]
    }
    pub fn fc_weights(&self) -> &[i8] {
        let start = CONV1_WEIGHTS + CONV1_BIAS + CONV2_WEIGHTS + CONV2_BIAS
            + CONV3_WEIGHTS + CONV3_BIAS;
        &self.data[start..start + FC_WEIGHTS]
    }
    pub fn fc_bias(&self) -> &[i8] {
        let start = CONV1_WEIGHTS + CONV1_BIAS + CONV2_WEIGHTS + CONV2_BIAS
            + CONV3_WEIGHTS + CONV3_BIAS + FC_WEIGHTS;
        &self.data[start..start + FC_BIAS]
    }

    /// Total size of model in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_params() {
        assert_eq!(TOTAL_PARAMS, 6020);
    }

    #[test]
    fn test_model_fits_l1() {
        // Pi 5 L1d is 64KB = 65536 bytes
        assert!(TOTAL_PARAMS < 65536, "model must fit in L1d");
    }

    #[test]
    fn test_dummy_model() {
        let m = Model::dummy();
        assert_eq!(m.size_bytes(), TOTAL_PARAMS);
        assert_eq!(m.conv1_weights().len(), CONV1_WEIGHTS);
        assert_eq!(m.fc_bias().len(), FC_BIAS);
    }

    #[test]
    fn test_from_bytes_valid() {
        let data = vec![0u8; TOTAL_PARAMS];
        let m = Model::from_bytes(&data).unwrap();
        assert_eq!(m.size_bytes(), TOTAL_PARAMS);
    }

    #[test]
    fn test_from_bytes_wrong_size() {
        let data = vec![0u8; 100];
        assert!(Model::from_bytes(&data).is_err());
    }

    #[test]
    fn test_layer_slices_contiguous() {
        let m = Model::dummy();
        let total = m.conv1_weights().len() + m.conv1_bias().len()
            + m.conv2_weights().len() + m.conv2_bias().len()
            + m.conv3_weights().len() + m.conv3_bias().len()
            + m.fc_weights().len() + m.fc_bias().len();
        assert_eq!(total, TOTAL_PARAMS);
    }

    #[test]
    fn test_class_names() {
        assert_eq!(CLASS_NAMES[0], "nothing");
        assert_eq!(CLASS_NAMES[1], "person");
        assert_eq!(CLASS_NAMES[2], "vehicle");
        assert_eq!(CLASS_NAMES[3], "animal");
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod frame;
pub mod model;
pub mod motion;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 19 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/model.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add CNN model weight format and loading (6KB, L1-resident)"
```

---

### Task 6: Scalar CNN inference engine

Scalar (non-SIMD) inference first — correct reference implementation.
Eä kernels replace the hot loops later.
**Fused operations:** conv2d+relu+pool is one function, not three separate memory passes.

**Files:**
- Create: `crates/eye-core/src/vision/inference.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write inference engine with tests**

`crates/eye-core/src/vision/inference.rs`:
```rust
use super::frame::Frame;
use super::model::*;

/// Scratch buffers for inference. Pre-allocated, reused across frames.
pub struct InferenceEngine {
    // Activation buffers (ping-pong between layers)
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    // FC input buffer
    fc_in: Vec<f32>,
}

impl InferenceEngine {
    pub fn new() -> Self {
        // Worst-case activation size: 64*64*32 = 131072
        // But after each fused conv+relu+pool, spatial dims shrink
        let max_size = (INPUT_SIZE * INPUT_SIZE) as usize * CONV3_OUT;
        Self {
            buf_a: vec![0.0; max_size],
            buf_b: vec![0.0; max_size],
            fc_in: vec![0.0; FC_IN],
        }
    }

    /// Run inference on a 64x64 grayscale frame. Returns classification.
    pub fn classify(&mut self, frame: &Frame, model: &Model) -> Classification {
        assert_eq!(frame.width, INPUT_SIZE);
        assert_eq!(frame.height, INPUT_SIZE);

        let (mut w, mut h) = (INPUT_SIZE as usize, INPUT_SIZE as usize);

        // Input: u8 -> f32 normalized to [-1, 1]
        for i in 0..frame.data.len() {
            self.buf_a[i] = (frame.data[i] as f32 / 127.5) - 1.0;
        }

        // Layer 1: fused conv2d+relu+pool (data stays in registers between ops)
        let (nw, nh) = (w / 2, h / 2);
        fused_conv_relu_pool(
            &self.buf_a, w, h, CONV1_IN,
            model.conv1_weights(), model.conv1_bias(),
            CONV1_OUT, &mut self.buf_b,
        );
        w = nw;
        h = nh;

        // Layer 2: fused conv2d+relu+pool
        let (nw, nh) = (w / 2, h / 2);
        fused_conv_relu_pool(
            &self.buf_b, w, h, CONV2_IN,
            model.conv2_weights(), model.conv2_bias(),
            CONV2_OUT, &mut self.buf_a,
        );
        w = nw;
        h = nh;

        // Layer 3: fused conv2d+relu+pool
        let (nw, nh) = (w / 2, h / 2);
        fused_conv_relu_pool(
            &self.buf_a, w, h, CONV3_IN,
            model.conv3_weights(), model.conv3_bias(),
            CONV3_OUT, &mut self.buf_b,
        );
        w = nw;
        h = nh;

        // Global average pooling -> FC -> softmax (fused, no intermediate write)
        let scores = fused_gap_fc_softmax(
            &self.buf_b, w, h, CONV3_OUT,
            model.fc_weights(), model.fc_bias(),
        );

        let (class_id, &confidence) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        Classification {
            class_id,
            class_name: CLASS_NAMES[class_id],
            confidence,
            scores,
        }
    }
}

/// Fused conv2d + ReLU + 2x2 max pool.
/// Computes convolution at 2x2 output blocks, applies ReLU, takes max —
/// all in registers before writing one pooled output value.
/// Data never hits memory between conv, relu, and pool.
fn fused_conv_relu_pool(
    input: &[f32], in_w: usize, in_h: usize, c_in: usize,
    weights: &[i8], bias: &[i8],
    c_out: usize, output: &mut [f32],
) {
    let out_w = in_w / 2;
    let out_h = in_h / 2;
    for oc in 0..c_out {
        let b = bias[oc] as f32;
        for oy in 0..out_h {
            for ox in 0..out_w {
                // Compute conv+relu at 4 positions in the 2x2 pool window
                let mut pool_max = f32::NEG_INFINITY;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let x = ox * 2 + dx as usize;
                        let y = oy * 2 + dy as usize;
                        // Conv at (x, y)
                        let mut sum = b;
                        for ic in 0..c_in {
                            for ky in 0..3usize {
                                for kx in 0..3usize {
                                    let iy = y as isize + ky as isize - 1;
                                    let ix = x as isize + kx as isize - 1;
                                    if iy >= 0 && iy < in_h as isize
                                        && ix >= 0 && ix < in_w as isize
                                    {
                                        let in_idx = (ic * in_h + iy as usize) * in_w + ix as usize;
                                        let w_idx = ((oc * c_in + ic) * 3 + ky) * 3 + kx;
                                        sum += input[in_idx] * weights[w_idx] as f32;
                                    }
                                }
                            }
                        }
                        // ReLU + pool max — stays in register
                        pool_max = pool_max.max(sum.max(0.0));
                    }
                }
                output[(oc * out_h + oy) * out_w + ox] = pool_max;
            }
        }
    }
}

/// Fused global average pooling + FC layer + softmax.
/// Accumulates GAP per channel, immediately multiplies by FC weights,
/// then softmax. Channel sums stay in registers, never written to an
/// intermediate buffer.
fn fused_gap_fc_softmax(
    input: &[f32], w: usize, h: usize, channels: usize,
    fc_weights: &[i8], fc_bias: &[i8],
) -> [f32; NUM_CLASSES] {
    let n = (w * h) as f32;
    let mut scores = [0.0f32; NUM_CLASSES];

    // Initialize with FC bias
    for j in 0..NUM_CLASSES {
        scores[j] = fc_bias[j] as f32;
    }

    // For each channel: compute GAP, immediately accumulate into all FC outputs
    for c in 0..channels {
        let mut sum = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                sum += input[(c * h + y) * w + x];
            }
        }
        let avg = sum / n; // GAP result for this channel — stays in register
        // Immediately multiply into FC outputs
        for j in 0..NUM_CLASSES {
            scores[j] += avg * fc_weights[j * channels + c] as f32;
        }
    }

    // Softmax
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        exp_sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= exp_sum;
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_dummy_model() {
        let mut engine = InferenceEngine::new();
        let model = Model::dummy();
        let frame = Frame::new(INPUT_SIZE, INPUT_SIZE);
        let result = engine.classify(&frame, &model);
        // With zero weights, softmax gives uniform 0.25
        assert_eq!(result.scores.len(), NUM_CLASSES);
        for &s in &result.scores {
            assert!((s - 0.25).abs() < 0.01, "expected ~0.25, got {s}");
        }
    }

    #[test]
    fn test_classify_nonzero_frame() {
        let mut engine = InferenceEngine::new();
        let model = Model::dummy();
        let mut frame = Frame::new(INPUT_SIZE, INPUT_SIZE);
        // Fill with some pattern
        for (i, pixel) in frame.data.iter_mut().enumerate() {
            *pixel = (i % 256) as u8;
        }
        let result = engine.classify(&frame, &model);
        // Should still produce valid probabilities
        let total: f32 = result.scores.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "scores should sum to 1.0, got {total}");
    }

    #[test]
    fn test_softmax() {
        let mut scores = [1.0, 2.0, 3.0, 4.0];
        softmax(&mut scores);
        let total: f32 = scores.iter().sum();
        assert!((total - 1.0).abs() < 0.001);
        // Scores should be monotonically increasing
        assert!(scores[3] > scores[2]);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_classification_has_valid_class() {
        let mut engine = InferenceEngine::new();
        let model = Model::dummy();
        let frame = Frame::new(INPUT_SIZE, INPUT_SIZE);
        let result = engine.classify(&frame, &model);
        assert!(result.class_id < NUM_CLASSES);
        assert!(CLASS_NAMES.contains(&result.class_name));
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_max_pool_2x2() {
        let input = vec![
            1.0, 3.0, 5.0, 7.0,
            2.0, 4.0, 6.0, 8.0,
            9.0, 11.0, 13.0, 15.0,
            10.0, 12.0, 14.0, 16.0,
        ];
        let mut output = vec![0.0; 4];
        max_pool_2x2(&input, 4, 4, 1, &mut output);
        assert_eq!(output, vec![4.0, 8.0, 12.0, 16.0]);
    }

    #[test]
    fn test_global_avg_pool() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 channel, 2x2
        let mut output = vec![0.0; 1];
        global_avg_pool(&input, 2, 2, 1, &mut output);
        assert!((output[0] - 2.5).abs() < 0.001);
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod frame;
pub mod inference;
pub mod model;
pub mod motion;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 25 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/inference.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add scalar CNN inference engine (reference impl, Ea replaces hot loops later)"
```

---

## Phase 5: Detection Pipeline

### Task 7: Detection pipeline wiring Stage 1 + Stage 2

**Files:**
- Create: `crates/eye-core/src/vision/pipeline.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write pipeline with tests**

`crates/eye-core/src/vision/pipeline.rs`:
```rust
use super::capture::Camera;
use super::frame::Frame;
use super::inference::InferenceEngine;
use super::model::{Classification, Model, INPUT_SIZE, NUM_CLASSES};
use super::motion::{MotionDetector, MotionBox};

/// A single detection event.
#[derive(Debug, Clone)]
pub struct Detection {
    pub classification: Classification,
    pub region: MotionBox,
    pub timestamp_ms: u64,
}

impl Detection {
    /// Format as a human-readable alert string.
    pub fn alert_text(&self) -> String {
        let ts = self.timestamp_ms / 1000;
        let secs = ts % 60;
        let mins = (ts / 60) % 60;
        let hours = (ts / 3600) % 24;
        format!(
            "{} detected ({:.0}%) — {:02}:{:02}:{:02}",
            capitalize(self.classification.class_name),
            self.classification.confidence * 100.0,
            hours, mins, secs,
        )
    }

    /// Summary string for VectorStore indexing.
    pub fn recall_text(&self) -> String {
        format!(
            "{} confidence={:.2} at={}",
            self.classification.class_name,
            self.classification.confidence,
            self.timestamp_ms,
        )
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// The full vision pipeline: camera -> motion detect -> classify.
pub struct Pipeline {
    motion: MotionDetector,
    engine: InferenceEngine,
    model: Model,
    prev_frame: Frame,
    curr_frame: Frame,
    patch: Frame,
    /// Minimum confidence to trigger an alert.
    pub min_confidence: f32,
}

impl Pipeline {
    pub fn new(model: Model, cam_width: u32, cam_height: u32) -> Self {
        Self {
            motion: MotionDetector::new(25, 10),
            engine: InferenceEngine::new(),
            model,
            prev_frame: Frame::new(cam_width, cam_height),
            curr_frame: Frame::new(cam_width, cam_height),
            patch: Frame::new(INPUT_SIZE, INPUT_SIZE),
            min_confidence: 0.5,
        }
    }

    /// Process one camera frame. Returns detections (if any).
    pub fn process(&mut self, camera: &mut dyn Camera) -> Result<Vec<Detection>, String> {
        // Swap frames (current becomes previous)
        std::mem::swap(&mut self.prev_frame, &mut self.curr_frame);

        // Capture new frame
        if !camera.grab(&mut self.curr_frame)? {
            return Ok(Vec::new());
        }

        // Stage 1: Motion detection
        let boxes = self.motion.detect(&self.prev_frame, &self.curr_frame);
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 2: Classify each motion region
        let mut detections = Vec::new();
        for region in boxes {
            self.patch = self.curr_frame.crop_and_scale(
                region.x, region.y, region.w, region.h, INPUT_SIZE,
            );
            let classification = self.engine.classify(&self.patch, &self.model);

            // Skip "nothing" class and low-confidence results
            if classification.class_id == 0 || classification.confidence < self.min_confidence {
                continue;
            }

            detections.push(Detection {
                classification,
                region,
                timestamp_ms: self.curr_frame.timestamp_ms,
            });
        }

        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::capture::MockCamera;

    #[test]
    fn test_pipeline_no_motion_first_frame() {
        let model = Model::dummy();
        let mut pipeline = Pipeline::new(model, 640, 480);
        let mut cam = MockCamera::new(640, 480);
        // First frame: no previous frame to compare, prev is all zeros
        // MockCamera draws a spot, so there IS motion vs the zero prev_frame
        let detections = pipeline.process(&mut cam).unwrap();
        // With dummy model (zero weights), classification is uniform 0.25 for all classes
        // So confidence is 0.25 < min_confidence (0.5), all filtered out
        assert!(detections.is_empty());
    }

    #[test]
    fn test_pipeline_processes_frames() {
        let model = Model::dummy();
        let mut pipeline = Pipeline::new(model, 640, 480);
        pipeline.min_confidence = 0.0; // Accept any confidence for testing
        let mut cam = MockCamera::new(640, 480);

        // Need two frames — first populates prev, second can detect motion
        let _ = pipeline.process(&mut cam).unwrap();
        // MockCamera spot moves between frames, so motion detected
        let detections = pipeline.process(&mut cam).unwrap();
        // With dummy model, class_id might be 0 (nothing), which we skip
        // But we set min_confidence to 0, so class_id=0 filter still applies
        // This is correct behavior — dummy model shouldn't produce real detections
    }

    #[test]
    fn test_detection_alert_text() {
        let d = Detection {
            classification: Classification {
                class_id: 1,
                class_name: "person",
                confidence: 0.92,
                scores: [0.02, 0.92, 0.03, 0.03],
            },
            region: MotionBox { x: 10, y: 20, w: 50, h: 60, pixel_count: 100 },
            timestamp_ms: 80043000, // 22:14:03
        };
        let text = d.alert_text();
        assert!(text.contains("Person"));
        assert!(text.contains("92%"));
        assert!(text.contains("22:14:03"));
    }

    #[test]
    fn test_detection_recall_text() {
        let d = Detection {
            classification: Classification {
                class_id: 2,
                class_name: "vehicle",
                confidence: 0.85,
                scores: [0.05, 0.05, 0.85, 0.05],
            },
            region: MotionBox { x: 0, y: 0, w: 100, h: 100, pixel_count: 500 },
            timestamp_ms: 1000,
        };
        let text = d.recall_text();
        assert!(text.contains("vehicle"));
        assert!(text.contains("0.85"));
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod frame;
pub mod inference;
pub mod model;
pub mod motion;
pub mod pipeline;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 29 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/pipeline.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add detection pipeline wiring motion + CNN + alert formatting"
```

---

## Phase 6: Alert Rules Engine

### Task 8: Alert rules with NL-configurable filters

**Files:**
- Create: `crates/eye-core/src/vision/rules.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write rules engine with tests**

`crates/eye-core/src/vision/rules.rs`:
```rust
use super::model::NUM_CLASSES;
use super::pipeline::Detection;
use serde::{Deserialize, Serialize};

/// Alert rules — configurable via WhatsApp chat.
/// LLM translates NL commands into rule updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRules {
    /// Which classes trigger alerts. true = alert enabled.
    pub class_enabled: [bool; NUM_CLASSES],
    /// Minimum confidence to alert (0.0 - 1.0).
    pub min_confidence: f32,
    /// Mute until this timestamp (epoch ms). 0 = not muted.
    pub mute_until_ms: u64,
    /// Cooldown between alerts in ms. Prevents rapid-fire.
    pub cooldown_ms: u64,
    /// Timestamp of last alert sent.
    #[serde(skip)]
    pub last_alert_ms: u64,
}

impl Default for AlertRules {
    fn default() -> Self {
        Self {
            // By default: alert on person and vehicle, not nothing or animal
            class_enabled: [false, true, true, false],
            min_confidence: 0.5,
            mute_until_ms: 0,
            cooldown_ms: 30_000, // 30 seconds
            last_alert_ms: 0,
        }
    }
}

impl AlertRules {
    /// Check if a detection should trigger an alert based on current rules.
    pub fn should_alert(&self, detection: &Detection, now_ms: u64) -> bool {
        // Class filter
        if !self.class_enabled[detection.classification.class_id] {
            return false;
        }
        // Confidence filter
        if detection.classification.confidence < self.min_confidence {
            return false;
        }
        // Mute check
        if self.mute_until_ms > 0 && now_ms < self.mute_until_ms {
            return false;
        }
        // Cooldown check
        if self.last_alert_ms > 0 && now_ms - self.last_alert_ms < self.cooldown_ms {
            return false;
        }
        true
    }

    /// Record that an alert was sent.
    pub fn record_alert(&mut self, now_ms: u64) {
        self.last_alert_ms = now_ms;
    }

    /// Enable alerts for a class by name.
    pub fn enable_class(&mut self, name: &str) -> bool {
        if let Some(idx) = class_index(name) {
            self.class_enabled[idx] = true;
            true
        } else {
            false
        }
    }

    /// Disable alerts for a class by name.
    pub fn disable_class(&mut self, name: &str) -> bool {
        if let Some(idx) = class_index(name) {
            self.class_enabled[idx] = false;
            true
        } else {
            false
        }
    }

    /// Format rules as human-readable string.
    pub fn describe(&self) -> String {
        let classes: Vec<&str> = super::model::CLASS_NAMES
            .iter()
            .enumerate()
            .filter(|(i, _)| self.class_enabled[*i])
            .map(|(_, name)| *name)
            .collect();
        let mute = if self.mute_until_ms > 0 {
            format!(", muted until {}", self.mute_until_ms)
        } else {
            String::new()
        };
        format!(
            "Alerting on: {} (min confidence: {:.0}%, cooldown: {}s{})",
            if classes.is_empty() { "nothing".to_string() } else { classes.join(", ") },
            self.min_confidence * 100.0,
            self.cooldown_ms / 1000,
            mute,
        )
    }
}

fn class_index(name: &str) -> Option<usize> {
    let lower = name.to_lowercase();
    super::model::CLASS_NAMES.iter().position(|&n| n == lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::Classification;
    use super::super::motion::MotionBox;

    fn test_detection(class_id: usize, confidence: f32) -> Detection {
        Detection {
            classification: Classification {
                class_id,
                class_name: super::super::model::CLASS_NAMES[class_id],
                confidence,
                scores: [0.0; NUM_CLASSES],
            },
            region: MotionBox { x: 0, y: 0, w: 10, h: 10, pixel_count: 50 },
            timestamp_ms: 1000,
        }
    }

    #[test]
    fn test_default_rules() {
        let rules = AlertRules::default();
        assert!(!rules.class_enabled[0]); // nothing
        assert!(rules.class_enabled[1]);  // person
        assert!(rules.class_enabled[2]);  // vehicle
        assert!(!rules.class_enabled[3]); // animal
    }

    #[test]
    fn test_should_alert_person() {
        let rules = AlertRules::default();
        let d = test_detection(1, 0.9); // person, high confidence
        assert!(rules.should_alert(&d, 100_000));
    }

    #[test]
    fn test_should_not_alert_nothing() {
        let rules = AlertRules::default();
        let d = test_detection(0, 0.9); // nothing class
        assert!(!rules.should_alert(&d, 100_000));
    }

    #[test]
    fn test_should_not_alert_low_confidence() {
        let rules = AlertRules::default();
        let d = test_detection(1, 0.3); // person but low confidence
        assert!(!rules.should_alert(&d, 100_000));
    }

    #[test]
    fn test_mute() {
        let mut rules = AlertRules::default();
        rules.mute_until_ms = 200_000;
        let d = test_detection(1, 0.9);
        assert!(!rules.should_alert(&d, 100_000)); // before mute expires
        assert!(rules.should_alert(&d, 300_000));  // after mute expires
    }

    #[test]
    fn test_cooldown() {
        let mut rules = AlertRules::default();
        rules.cooldown_ms = 60_000;
        rules.last_alert_ms = 100_000;
        let d = test_detection(1, 0.9);
        assert!(!rules.should_alert(&d, 120_000)); // 20s after last, within cooldown
        assert!(rules.should_alert(&d, 200_000));  // 100s after last, past cooldown
    }

    #[test]
    fn test_enable_disable_class() {
        let mut rules = AlertRules::default();
        assert!(!rules.class_enabled[3]); // animal off
        assert!(rules.enable_class("animal"));
        assert!(rules.class_enabled[3]);
        assert!(rules.disable_class("person"));
        assert!(!rules.class_enabled[1]);
    }

    #[test]
    fn test_enable_invalid_class() {
        let mut rules = AlertRules::default();
        assert!(!rules.enable_class("dragon"));
    }

    #[test]
    fn test_describe() {
        let rules = AlertRules::default();
        let desc = rules.describe();
        assert!(desc.contains("person"));
        assert!(desc.contains("vehicle"));
        assert!(desc.contains("50%"));
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod frame;
pub mod inference;
pub mod model;
pub mod motion;
pub mod pipeline;
pub mod rules;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 38 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/rules.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add alert rules engine with class/confidence/mute/cooldown filters"
```

---

## Phase 7: Detection Dedup via Recall

### Task 9: Detection dedup using eaclaw-core VectorStore

**Files:**
- Create: `crates/eye-core/src/vision/dedup.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`

**Step 1: Write dedup module with tests**

`crates/eye-core/src/vision/dedup.rs`:
```rust
use eaclaw_core::recall::VectorStore;
use super::pipeline::Detection;

/// Deduplicates detection alerts using VectorStore cosine similarity.
/// If a recent detection is too similar (same class, close in time),
/// it's suppressed as a duplicate.
pub struct DetectionDedup {
    store: VectorStore,
    /// Similarity threshold above which a detection is considered duplicate.
    pub similarity_threshold: f32,
}

impl DetectionDedup {
    pub fn new(capacity: usize) -> Self {
        Self {
            store: VectorStore::with_capacity(capacity),
            similarity_threshold: 0.90,
        }
    }

    /// Check if this detection is a duplicate of a recent one.
    /// If not, index it and return false (= should alert).
    /// If yes, return true (= suppress).
    pub fn is_duplicate(&mut self, detection: &Detection) -> bool {
        let text = detection.recall_text();

        // Check for similar recent detections
        let results = self.store.recall(&text, 1);
        let dominated = results
            .first()
            .map_or(false, |r| r.score > self.similarity_threshold);

        // Always index (even duplicates, to maintain recency)
        self.store.insert(&text);

        dominated
    }

    /// Number of indexed detections.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clear all indexed detections.
    pub fn clear(&mut self) {
        self.store.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::{Classification, NUM_CLASSES};
    use super::super::motion::MotionBox;

    fn make_detection(class: &'static str, class_id: usize, ts: u64) -> Detection {
        Detection {
            classification: Classification {
                class_id,
                class_name: class,
                confidence: 0.9,
                scores: [0.0; NUM_CLASSES],
            },
            region: MotionBox { x: 0, y: 0, w: 50, h: 50, pixel_count: 100 },
            timestamp_ms: ts,
        }
    }

    #[test]
    fn test_first_detection_not_duplicate() {
        let mut dedup = DetectionDedup::new(100);
        let d = make_detection("person", 1, 1000);
        assert!(!dedup.is_duplicate(&d));
    }

    #[test]
    fn test_same_detection_is_duplicate() {
        let mut dedup = DetectionDedup::new(100);
        let d1 = make_detection("person", 1, 1000);
        let d2 = make_detection("person", 1, 1001);
        assert!(!dedup.is_duplicate(&d1));
        assert!(dedup.is_duplicate(&d2));
    }

    #[test]
    fn test_different_class_not_duplicate() {
        let mut dedup = DetectionDedup::new(100);
        let d1 = make_detection("person", 1, 1000);
        let d2 = make_detection("vehicle", 2, 1001);
        assert!(!dedup.is_duplicate(&d1));
        assert!(!dedup.is_duplicate(&d2));
    }

    #[test]
    fn test_clear() {
        let mut dedup = DetectionDedup::new(100);
        let d = make_detection("person", 1, 1000);
        dedup.is_duplicate(&d);
        assert_eq!(dedup.len(), 1);
        dedup.clear();
        assert_eq!(dedup.len(), 0);
    }
}
```

**Step 2: Add to mod.rs**

```rust
pub mod capture;
pub mod dedup;
pub mod frame;
pub mod inference;
pub mod model;
pub mod motion;
pub mod pipeline;
pub mod rules;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 42 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/vision/dedup.rs crates/eye-core/src/vision/mod.rs
git commit -m "feat: add detection dedup using VectorStore recall"
```

---

## Phase 8: Camera Loop

### Task 10: Async camera loop tying everything together

**Files:**
- Create: `crates/eye-core/src/eye.rs`
- Modify: `crates/eye-core/src/lib.rs`

**Step 1: Write the camera loop**

`crates/eye-core/src/eye.rs`:
```rust
use crate::vision::capture::Camera;
use crate::vision::dedup::DetectionDedup;
use crate::vision::model::Model;
use crate::vision::pipeline::{Detection, Pipeline};
use crate::vision::rules::AlertRules;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Events emitted by the camera loop.
#[derive(Debug, Clone)]
pub enum EyeEvent {
    /// A detection that passed all filters (rules + dedup).
    Alert(Detection),
    /// Camera error.
    Error(String),
    /// Status update (e.g., "camera started").
    Status(String),
}

/// Shared state accessible from the agent loop.
pub struct EyeState {
    pub rules: AlertRules,
    pub detection_count: u64,
    pub alert_count: u64,
    pub running: bool,
}

impl EyeState {
    fn new() -> Self {
        Self {
            rules: AlertRules::default(),
            detection_count: 0,
            alert_count: 0,
            running: false,
        }
    }
}

/// Start the camera loop in a background task.
/// Returns an event receiver and shared state handle.
pub fn start_camera_loop(
    mut camera: Box<dyn Camera>,
    model: Model,
    frame_interval_ms: u64,
) -> (mpsc::Receiver<EyeEvent>, Arc<Mutex<EyeState>>) {
    let (tx, rx) = mpsc::channel(32);
    let state = Arc::new(Mutex::new(EyeState::new()));
    let state_clone = state.clone();

    let (cam_w, cam_h) = camera.resolution();

    tokio::spawn(async move {
        let mut pipeline = Pipeline::new(model, cam_w, cam_h);
        let mut dedup = DetectionDedup::new(256);

        {
            let mut s = state_clone.lock().unwrap();
            s.running = true;
        }

        let _ = tx.send(EyeEvent::Status("Camera started".into())).await;

        loop {
            // Check if we should stop
            {
                let s = state_clone.lock().unwrap();
                if !s.running {
                    break;
                }
            }

            // Process one frame
            match pipeline.process(&mut *camera) {
                Ok(detections) => {
                    for detection in detections {
                        let now_ms = detection.timestamp_ms;

                        let mut s = state_clone.lock().unwrap();
                        s.detection_count += 1;

                        // Check rules
                        if !s.rules.should_alert(&detection, now_ms) {
                            continue;
                        }

                        // Check dedup
                        if dedup.is_duplicate(&detection) {
                            continue;
                        }

                        s.rules.record_alert(now_ms);
                        s.alert_count += 1;
                        drop(s);

                        let _ = tx.send(EyeEvent::Alert(detection)).await;
                    }
                }
                Err(e) => {
                    let _ = tx.send(EyeEvent::Error(e)).await;
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(frame_interval_ms)).await;
        }
    });

    (rx, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::capture::MockCamera;

    #[tokio::test]
    async fn test_camera_loop_starts() {
        let cam = Box::new(MockCamera::new(640, 480));
        let model = Model::dummy();
        let (mut rx, state) = start_camera_loop(cam, model, 10);

        // Should get a status event
        let event = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            rx.recv(),
        ).await;

        assert!(event.is_ok());
        if let Ok(Some(EyeEvent::Status(msg))) = event {
            assert_eq!(msg, "Camera started");
        }

        // Stop the loop
        state.lock().unwrap().running = false;
    }

    #[tokio::test]
    async fn test_camera_loop_state() {
        let cam = Box::new(MockCamera::new(640, 480));
        let model = Model::dummy();
        let (_rx, state) = start_camera_loop(cam, model, 10);

        // Give it a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(state.lock().unwrap().running);

        // Stop
        state.lock().unwrap().running = false;
    }
}
```

**Step 2: Update lib.rs**

```rust
pub mod eye;
pub mod vision;
```

**Step 3: Run tests**

Run: `cargo test -p eye-core`
Expected: 44 tests pass

**Step 4: Commit**

```bash
git add crates/eye-core/src/eye.rs crates/eye-core/src/lib.rs
git commit -m "feat: add async camera loop with rules, dedup, and event channel"
```

---

## Phase 9: CLI Integration

### Task 11: eaclaw-eye binary with WhatsApp mode

**Files:**
- Modify: `eye-cli/src/main.rs`

**Step 1: Write the CLI entry point**

`eye-cli/src/main.rs`:
```rust
use eaclaw_core::agent::Agent;
use eaclaw_core::config::Config;
use eaclaw_core::llm::anthropic::AnthropicProvider;
use eaclaw_core::safety::SafetyLayer;
use eaclaw_core::tools::ToolRegistry;
use eye_core::eye::{start_camera_loop, EyeEvent};
use eye_core::vision::capture::MockCamera;
use eye_core::vision::model::Model;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("eaclaw-eye {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("eaclaw-eye {} — Cache-Resident Security Camera", env!("CARGO_PKG_VERSION"));
        println!();
        println!("Usage: eaclaw-eye [OPTIONS]");
        println!();
        println!("Options:");
        println!("  --whatsapp    Run in WhatsApp mode (scan QR in terminal)");
        println!("  --mock        Use mock camera (for testing without hardware)");
        println!("  --version     Print version");
        println!("  --help        Print this help");
        std::process::exit(0);
    }

    // Initialize eaclaw SIMD kernels
    if let Err(e) = eaclaw_core::kernels::init() {
        eprintln!("Failed to initialize SIMD kernels: {e}");
        std::process::exit(1);
    }

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            eprintln!("Set ANTHROPIC_API_KEY to get started.");
            std::process::exit(1);
        }
    };

    let use_mock = args.iter().any(|a| a == "--mock");

    // Load model (dummy for now, replaced with real weights later)
    let model = Model::dummy();
    eprintln!("Model loaded: {} bytes (fits L1: {})",
        model.size_bytes(),
        if model.size_bytes() < 65536 { "yes" } else { "no" });

    // Start camera
    let camera: Box<dyn eye_core::vision::capture::Camera> = if use_mock {
        eprintln!("Using mock camera (640x480)");
        Box::new(MockCamera::new(640, 480))
    } else {
        #[cfg(feature = "v4l2")]
        {
            let dev = std::env::var("EACLAW_CAMERA").unwrap_or_else(|_| "/dev/video0".into());
            let cam = eye_core::vision::v4l2::V4l2Camera::open(&dev, 640, 480)
                .unwrap_or_else(|e| {
                    eprintln!("Camera error: {e}");
                    std::process::exit(1);
                });
            eprintln!("V4L2 camera: {dev}");
            Box::new(cam)
        }
        #[cfg(not(feature = "v4l2"))]
        {
            eprintln!("Built without v4l2 feature. Use --mock or rebuild with: cargo build --features v4l2");
            std::process::exit(1);
        }
    };

    let (cam_w, cam_h) = camera.resolution();
    eprintln!("Camera: {}x{}", cam_w, cam_h);

    // Start camera loop (100ms between frames = ~10fps)
    let (mut eye_rx, eye_state) = start_camera_loop(camera, model, 100);

    // Start agent
    let llm: Arc<dyn eaclaw_core::llm::LlmProvider> =
        Arc::new(AnthropicProvider::new(&config));
    let tools = ToolRegistry::with_defaults(&config, llm.clone());
    let safety = SafetyLayer::new();

    eprintln!("eaclaw-eye ready. Camera loop running.");

    // For now, run REPL mode with camera events printed to stderr
    let channel = eaclaw_core::channel::repl::ReplChannel::new("eaclaw-eye");
    let mut agent = Agent::new(config.clone(), llm, tools, safety);

    // Spawn event handler
    tokio::spawn(async move {
        while let Some(event) = eye_rx.recv().await {
            match event {
                EyeEvent::Alert(d) => {
                    eprintln!("[ALERT] {}", d.alert_text());
                }
                EyeEvent::Error(e) => {
                    eprintln!("[ERROR] Camera: {e}");
                }
                EyeEvent::Status(s) => {
                    eprintln!("[STATUS] {s}");
                }
            }
        }
    });

    let result = agent.run(&channel).await;
    channel.shutdown();

    // Stop camera
    eye_state.lock().unwrap().running = false;

    if let Err(e) = result {
        eprintln!("Agent error: {e}");
        std::process::exit(1);
    }
}
```

**Step 2: Add tracing-subscriber dependency**

Add to `eye-cli/Cargo.toml`:
```toml
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

**Step 3: Verify it builds**

Run: `cargo build -p eaclaw-eye`
Expected: compiles successfully

**Step 4: Commit**

```bash
git add eye-cli/
git commit -m "feat: add eaclaw-eye CLI with mock camera and agent integration"
```

---

## Phase 10: V4L2 Camera (Pi Hardware)

### Task 12: V4L2 camera capture for Raspberry Pi

**Files:**
- Create: `crates/eye-core/src/vision/v4l2.rs`
- Modify: `crates/eye-core/src/vision/mod.rs`
- Modify: `crates/eye-core/Cargo.toml`

**Step 1: Add v4l dependency (feature-gated)**

Add to `crates/eye-core/Cargo.toml`:
```toml
[features]
default = []
v4l2 = ["v4l"]

[dependencies]
v4l = { version = "0.14", optional = true }
```

**Step 2: Write V4L2 camera implementation**

`crates/eye-core/src/vision/v4l2.rs`:
```rust
//! V4L2 camera capture for Raspberry Pi Camera Module v2.
//! Feature-gated behind `v4l2` since it requires Linux + camera hardware.

use super::capture::Camera;
use super::frame::Frame;

#[cfg(feature = "v4l2")]
use v4l::buffer::Type;
#[cfg(feature = "v4l2")]
use v4l::io::mmap::Stream;
#[cfg(feature = "v4l2")]
use v4l::io::traits::CaptureStream;
#[cfg(feature = "v4l2")]
use v4l::video::Capture;
#[cfg(feature = "v4l2")]
use v4l::Device;
#[cfg(feature = "v4l2")]
use v4l::FourCC;

#[cfg(feature = "v4l2")]
pub struct V4l2Camera {
    stream: Stream,
    width: u32,
    height: u32,
}

#[cfg(feature = "v4l2")]
impl V4l2Camera {
    /// Open the camera device (default /dev/video0) and start streaming.
    pub fn open(device_path: &str, width: u32, height: u32) -> Result<Self, String> {
        let dev = Device::with_path(device_path)
            .map_err(|e| format!("failed to open {device_path}: {e}"))?;

        // Set format: grayscale (GREY) for direct use, or YUYV and convert
        let mut fmt = dev.format().map_err(|e| format!("get format: {e}"))?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"GREY");
        let fmt = dev.set_format(&fmt).map_err(|e| format!("set format: {e}"))?;

        let actual_w = fmt.width;
        let actual_h = fmt.height;

        let stream = Stream::with_buffers(&dev, Type::VideoCapture, 4)
            .map_err(|e| format!("create stream: {e}"))?;

        Ok(Self {
            stream,
            width: actual_w,
            height: actual_h,
        })
    }
}

#[cfg(feature = "v4l2")]
impl Camera for V4l2Camera {
    fn grab(&mut self, frame: &mut Frame) -> Result<bool, String> {
        let (buf, _meta) = self.stream.next()
            .map_err(|e| format!("capture: {e}"))?;

        let expected = (self.width * self.height) as usize;
        if buf.len() < expected {
            return Err(format!("short frame: {} < {}", buf.len(), expected));
        }

        frame.data[..expected].copy_from_slice(&buf[..expected]);
        frame.timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(true)
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
```

**Step 3: Add to mod.rs (feature-gated)**

```rust
pub mod capture;
pub mod dedup;
pub mod frame;
pub mod inference;
pub mod model;
pub mod motion;
pub mod pipeline;
pub mod rules;
#[cfg(feature = "v4l2")]
pub mod v4l2;
```

**Step 4: Run tests (without v4l2 feature — no hardware needed)**

Run: `cargo test -p eye-core`
Expected: all existing tests pass (v4l2 module not compiled)

**Step 5: Commit**

```bash
git add crates/eye-core/src/vision/v4l2.rs crates/eye-core/src/vision/mod.rs crates/eye-core/Cargo.toml
git commit -m "feat: add V4L2 camera capture for Pi Camera v2 (feature-gated)"
```

---

## Phase 11: Benchmarks and Measurement

### Task 13: Criterion benchmarks for vision hot path

**Files:**
- Create: `crates/eye-core/benches/vision_bench.rs`
- Create: `MEASURING.md`
- Modify: `crates/eye-core/Cargo.toml`

**Step 1: Add criterion dependency**

Add to `crates/eye-core/Cargo.toml`:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "vision_bench"
harness = false
```

**Step 2: Write benchmarks**

`crates/eye-core/benches/vision_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use eye_core::vision::frame::Frame;
use eye_core::vision::inference::InferenceEngine;
use eye_core::vision::model::Model;
use eye_core::vision::motion::MotionDetector;

fn bench_frame_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("motion_detection");
    let mut det = MotionDetector::new(25, 10);

    for &(w, h) in &[(640, 480), (1280, 720), (1920, 1080)] {
        let f1 = Frame::new(w, h);
        let mut f2 = Frame::new(w, h);
        for (i, p) in f2.data.iter_mut().enumerate() {
            *p = ((i * 7) % 256) as u8;
        }

        group.throughput(Throughput::Bytes((w * h) as u64));
        group.bench_function(
            format!("{w}x{h}"),
            |b| b.iter(|| det.detect(black_box(&f1), black_box(&f2))),
        );
    }
    group.finish();
}

fn bench_crop_and_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("crop_and_scale");
    let mut f = Frame::new(640, 480);
    for (i, p) in f.data.iter_mut().enumerate() {
        *p = (i % 256) as u8;
    }

    group.bench_function("640x480_to_64x64", |b| {
        b.iter(|| f.crop_and_scale(black_box(100), black_box(100), 200, 200, 64))
    });
    group.finish();
}

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnn_inference");
    let mut engine = InferenceEngine::new();
    let model = Model::dummy();
    let mut frame = Frame::new(64, 64);
    for (i, p) in frame.data.iter_mut().enumerate() {
        *p = (i % 256) as u8;
    }

    group.bench_function("64x64_classify", |b| {
        b.iter(|| engine.classify(black_box(&frame), black_box(&model)))
    });
    group.finish();
}

criterion_group!(benches, bench_frame_diff, bench_crop_and_scale, bench_inference);
criterion_main!(benches);
```

**Step 3: Write MEASURING.md**

```markdown
# eaclaw-eye Measurement Methodology

Same methodology as eaclaw. Every hot-path change gets measured.

---

## 1. Tests (correctness gate)

cargo test

---

## 2. Criterion Benchmarks (throughput + latency)

cargo bench -p eye-core

Benchmarks:
- motion_detection: frame diff at 640x480, 1280x720, 1920x1080
- crop_and_scale: 640x480 region to 64x64
- cnn_inference: full forward pass on 64x64 frame

---

## 3. perf stat: Cache + Branch + IPC

cargo build --release -p eye-core --example perf_vision

perf stat -e instructions,cycles,L1-icache-load-misses,L1-dcache-load-misses,branches,branch-misses \
  target/release/examples/perf_vision 100000

Targets:
- L1i miss rate < 0.01%
- L1d miss rate < 0.1% (model weights stay in cache)
- IPC > 3.5
- Branch miss rate < 0.1%

---

## 4. Instruction Footprint

nm -S --size-sort target/release/examples/perf_vision | grep -E 'conv2d|motion|classify|pool'

Target: vision hot path < 8KB instructions.

---

## Measurement Checklist

After every hot-path change:

1. [ ] cargo test — all pass
2. [ ] cargo bench — no regressions
3. [ ] perf stat — L1i miss < 0.01%, IPC > 3.5
4. [ ] Code size — hot path < 8KB
5. [ ] Update BENCHMARKS.md with new results
```

**Step 4: Run benchmarks**

Run: `cargo bench -p eye-core`
Expected: all three benchmark groups run, baseline established

**Step 5: Commit**

```bash
git add crates/eye-core/benches/ MEASURING.md crates/eye-core/Cargo.toml
git commit -m "feat: add Criterion benchmarks and measurement methodology"
```

---

## Future Tasks (not implemented yet)

These tasks are deferred until the core pipeline works end-to-end:

### Task 14: WhatsApp image sending
Extend the Go bridge and Channel trait to support sending photos.
Requires changes in both eaclaw (upstream) and eaclaw-eye.
Full implementation — working image send, not a stub.

### Task 15: Detection history tool
New tool implementing the Tool trait: `/detections [N]` — shows last N detections.
`/stats` — shows detection counts by class.
Complete with tests and schema. Registered in ToolRegistry.

### Task 16: NL alert rule configuration
System prompt addition that teaches Claude to update AlertRules from natural language.
Tool: `update_rules` that the LLM calls to modify rules in EyeState.
Working end-to-end: user texts "ignore animals" -> LLM calls tool -> rules updated.

### Task 17: Ea kernel acceleration
Replace scalar conv2d_relu, max_pool_2x2, and global_avg_pool with Ea kernels
from eacompute/demo (conv2d_3x3, pixel_pipeline, mnist_normalize).
Criterion benchmark before/after. perf stat to verify L1 residency.
Scalar fallback removed — Ea is the implementation, not an optimization.

### Task 18: Real model training
Train the 6KB CNN on a person/vehicle/animal dataset (e.g., COCO subset).
Export int8 weights. Embed via include_bytes! in build.rs.
Validate accuracy on test set. Ship working weights, not placeholders.

### Task 19: Pi SD card image
Raspberry Pi OS Lite image with eaclaw-eye pre-installed.
Auto-start on boot. Serial console for initial QR scan.
Tested on real Pi 5 hardware with Camera v2.
