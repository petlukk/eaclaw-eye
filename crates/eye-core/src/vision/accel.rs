/// FFI bridge to eacompute SIMD kernels (.ea compiled to .so).
/// Feature-gated behind `ea` — falls back to scalar Rust when disabled.
///
/// Kernels are embedded in the binary and extracted to ~/.eaclaw/lib/eye-v{VERSION}/
/// on first run. Functions are loaded at runtime via libloading.
/// Call `init()` once at startup before using any kernel function.

use libloading::{Library, Symbol};
use std::path::PathBuf;
use std::sync::OnceLock;

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}

// ---- FFI function type aliases ----

type MotionFusedFn = unsafe extern "C" fn(
    *const u8, *const u8, *mut u32, i32, i32, i32, i32, i32,
) -> i32;

type NormalizeU8F32Fn = unsafe extern "C" fn(
    *const u8, *mut f32, i32, f32, f32,
);

type RelayoutConvWeightsFn = unsafe extern "C" fn(
    *const i8, *mut i8, i32, i32,
);

type Conv3x3ReluPoolFn = unsafe extern "C" fn(
    *const f32, i32, i32, i32, *const i8, *const i8, i32, *mut f32,
);

type GapFcSoftmaxFn = unsafe extern "C" fn(
    *const f32, i32, i32, i32, *const i8, *const i8, *mut f32, i32,
);

// ---- Kernel table ----

struct KernelTable {
    _libs: Vec<Library>,
    motion_fused: MotionFusedFn,
    normalize_u8_f32: NormalizeU8F32Fn,
    relayout_conv_weights: RelayoutConvWeightsFn,
    conv3x3_relu_pool: Conv3x3ReluPoolFn,
    gap_fc_softmax: GapFcSoftmaxFn,
}

// SAFETY: KernelTable holds function pointers and library handles.
// The function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded (held in OnceLock for program lifetime).
unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

static KERNELS: OnceLock<KernelTable> = OnceLock::new();

fn k() -> &'static KernelTable {
    KERNELS.get_or_init(|| {
        let lib_dir = extract_kernels().expect("failed to extract eye kernels");
        load_kernels(&lib_dir).expect("failed to load eye kernels")
    })
}

/// Initialize the kernel runtime: extract embedded .so files and load them.
/// Safe to call multiple times (only the first call does work).
pub fn init() -> Result<(), String> {
    if KERNELS.get().is_some() {
        return Ok(());
    }
    let lib_dir = extract_kernels()?;
    let table = load_kernels(&lib_dir)?;
    let _ = KERNELS.set(table);
    eprintln!("eye kernels: v{} loaded", embedded::VERSION);
    Ok(())
}

fn kernel_dir() -> Result<PathBuf, String> {
    let base = home::home_dir()
        .ok_or_else(|| "cannot determine home directory".to_string())?;
    Ok(base.join(".eaclaw").join("lib").join(format!("eye-v{}", embedded::VERSION)))
}

fn extract_kernels() -> Result<PathBuf, String> {
    let dir = kernel_dir()?;

    let marker = dir.join(".extracted");
    if marker.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let kernels: &[(&str, &[u8])] = &[
        ("libmotion.so", embedded::MOTION),
        ("libinference.so", embedded::INFERENCE),
        ("libnormalize.so", embedded::NORMALIZE),
    ];

    for (name, data) in kernels {
        let path = dir.join(name);
        std::fs::write(&path, data)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    }

    let _ = std::fs::write(&marker, embedded::VERSION);
    Ok(dir)
}

fn load_kernels(lib_dir: &PathBuf) -> Result<KernelTable, String> {
    let load = |name: &str| -> Result<Library, String> {
        let path = lib_dir.join(format!("lib{name}.so"));
        unsafe {
            Library::new(&path).map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    let motion = load("motion")?;
    let inference = load("inference")?;
    let normalize = load("normalize")?;

    unsafe {
        let sym = |lib: &Library, name: &[u8]| -> Result<usize, String> {
            let s: Symbol<*const ()> = lib.get(name)
                .map_err(|e| format!("symbol {:?}: {e}", std::str::from_utf8(name)))?;
            Ok(*s as usize)
        };

        let table = KernelTable {
            motion_fused: std::mem::transmute(
                sym(&motion, b"motion_fused\0")?),
            normalize_u8_f32: std::mem::transmute(
                sym(&normalize, b"normalize_u8_f32\0")?),
            relayout_conv_weights: std::mem::transmute(
                sym(&inference, b"relayout_conv_weights\0")?),
            conv3x3_relu_pool: std::mem::transmute(
                sym(&inference, b"conv3x3_relu_pool\0")?),
            gap_fc_softmax: std::mem::transmute(
                sym(&inference, b"gap_fc_softmax\0")?),
            _libs: vec![motion, inference, normalize],
        };
        Ok(table)
    }
}

// ---- Safe wrappers ----

/// Fused motion detection: abs-diff + threshold + grid counting.
/// Returns total number of changed pixels.
pub fn ea_motion_fused(
    prev: &[u8],
    curr: &[u8],
    grid_counts: &mut [u32],
    width: u32,
    height: u32,
    threshold: u8,
    cell_size: u32,
    cols: u32,
) -> u32 {
    debug_assert_eq!(prev.len(), (width * height) as usize);
    debug_assert_eq!(curr.len(), (width * height) as usize);
    debug_assert!(grid_counts.len() >= (cols * ((height + cell_size - 1) / cell_size)) as usize);

    let result = unsafe {
        (k().motion_fused)(
            prev.as_ptr(),
            curr.as_ptr(),
            grid_counts.as_mut_ptr(),
            width as i32,
            height as i32,
            threshold as i32,
            cell_size as i32,
            cols as i32,
        )
    };
    result as u32
}

/// Normalize u8 frame to f32: out[i] = input[i] * scale + bias.
/// For CNN input, use scale = 1.0/127.5, bias = -1.0.
pub fn ea_normalize_u8_f32(input: &[u8], out: &mut [f32], scale: f32, bias: f32) {
    debug_assert!(out.len() >= input.len());

    unsafe {
        (k().normalize_u8_f32)(
            input.as_ptr(),
            out.as_mut_ptr(),
            input.len() as i32,
            scale,
            bias,
        );
    }
}

/// Relayout conv weights from [c_out][c_in][3][3] to [c_out][3][3][c_in].
/// Called once at model load, not on the hot path.
pub fn ea_relayout_conv_weights(src: &[i8], dst: &mut [i8], c_out: usize, c_in: usize) {
    debug_assert_eq!(src.len(), c_out * c_in * 9);
    debug_assert_eq!(dst.len(), c_out * c_in * 9);

    unsafe {
        (k().relayout_conv_weights)(
            src.as_ptr(),
            dst.as_mut_ptr(),
            c_out as i32,
            c_in as i32,
        );
    }
}

/// Fused conv3x3 + bias + ReLU + 2x2 max-pool.
/// Weights must be in [c_out][3][3][c_in] layout (use ea_relayout_conv_weights first).
pub fn ea_conv3x3_relu_pool(
    input: &[f32],
    in_w: usize,
    in_h: usize,
    c_in: usize,
    weights: &[i8],
    bias: &[i8],
    c_out: usize,
    output: &mut [f32],
) {
    let out_w = in_w / 2;
    let out_h = in_h / 2;
    debug_assert!(input.len() >= in_w * in_h * c_in);
    debug_assert_eq!(weights.len(), c_out * c_in * 9);
    debug_assert_eq!(bias.len(), c_out);
    debug_assert!(output.len() >= out_w * out_h * c_out);

    unsafe {
        (k().conv3x3_relu_pool)(
            input.as_ptr(),
            in_w as i32,
            in_h as i32,
            c_in as i32,
            weights.as_ptr(),
            bias.as_ptr(),
            c_out as i32,
            output.as_mut_ptr(),
        );
    }
}

/// Fused global average pooling + fully-connected + softmax.
pub fn ea_gap_fc_softmax(
    input: &[f32],
    w: usize,
    h: usize,
    channels: usize,
    fc_weights: &[i8],
    fc_bias: &[i8],
    scores_out: &mut [f32],
    num_classes: usize,
) {
    debug_assert!(input.len() >= w * h * channels);
    debug_assert_eq!(fc_weights.len(), channels * num_classes);
    debug_assert_eq!(fc_bias.len(), num_classes);
    debug_assert!(scores_out.len() >= num_classes);

    unsafe {
        (k().gap_fc_softmax)(
            input.as_ptr(),
            w as i32,
            h as i32,
            channels as i32,
            fc_weights.as_ptr(),
            fc_bias.as_ptr(),
            scores_out.as_mut_ptr(),
            num_classes as i32,
        );
    }
}
