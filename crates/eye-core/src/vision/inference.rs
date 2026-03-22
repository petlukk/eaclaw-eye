/// Fused CNN inference engine. Conv2d+ReLU+MaxPool and GAP+FC+Softmax are
/// each a single fused pass -- intermediate conv/relu values never leave registers.

use super::frame::Frame;
use super::model::*;

pub struct InferenceEngine {
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    /// Relaid-out conv weights for EA kernels: [c_out][3][3][c_in] layout.
    /// Allocated once, reused across frames. Only populated when `ea` feature is active.
    #[cfg(feature = "ea")]
    conv1_weights_relaid: Vec<i8>,
    #[cfg(feature = "ea")]
    conv2_weights_relaid: Vec<i8>,
    #[cfg(feature = "ea")]
    conv3_weights_relaid: Vec<i8>,
    #[cfg(feature = "ea")]
    weights_ready: bool,
}

impl InferenceEngine {
    pub fn new() -> Self {
        let worst_case = (INPUT_SIZE as usize) * (INPUT_SIZE as usize) * CONV3_OUT;
        Self {
            buf_a: vec![0.0f32; worst_case],
            buf_b: vec![0.0f32; worst_case],
            #[cfg(feature = "ea")]
            conv1_weights_relaid: vec![0i8; CONV1_WEIGHTS],
            #[cfg(feature = "ea")]
            conv2_weights_relaid: vec![0i8; CONV2_WEIGHTS],
            #[cfg(feature = "ea")]
            conv3_weights_relaid: vec![0i8; CONV3_WEIGHTS],
            #[cfg(feature = "ea")]
            weights_ready: false,
        }
    }

    /// Relayout weights on first use (once per model, not per frame).
    #[cfg(feature = "ea")]
    fn ensure_weights_relaid(&mut self, model: &Model) {
        if self.weights_ready {
            return;
        }
        use super::accel::ea_relayout_conv_weights;
        ea_relayout_conv_weights(
            model.conv1_weights(), &mut self.conv1_weights_relaid,
            CONV1_OUT, CONV1_IN,
        );
        ea_relayout_conv_weights(
            model.conv2_weights(), &mut self.conv2_weights_relaid,
            CONV2_OUT, CONV2_IN,
        );
        ea_relayout_conv_weights(
            model.conv3_weights(), &mut self.conv3_weights_relaid,
            CONV3_OUT, CONV3_IN,
        );
        self.weights_ready = true;
    }

    pub fn classify(&mut self, frame: &Frame, model: &Model) -> Classification {
        let sz = INPUT_SIZE as usize;

        // Step 1: Normalize u8 -> f32 into buf_a (64x64x1)
        #[cfg(feature = "ea")]
        {
            let n = frame.data.len().min(sz * sz);
            super::accel::ea_normalize_u8_f32(
                &frame.data[..n], &mut self.buf_a[..n],
                1.0 / 127.5, -1.0,
            );
        }
        #[cfg(not(feature = "ea"))]
        {
            for i in 0..frame.data.len().min(sz * sz) {
                self.buf_a[i] = (frame.data[i] as f32 / 127.5) - 1.0;
            }
        }

        #[cfg(feature = "ea")]
        {
            use super::accel::{ea_conv3x3_relu_pool, ea_gap_fc_softmax};
            self.ensure_weights_relaid(model);

            // Layer 1: 64x64x1 -> 32x32x8
            ea_conv3x3_relu_pool(
                &self.buf_a, 64, 64, CONV1_IN,
                &self.conv1_weights_relaid, model.conv1_bias(),
                CONV1_OUT, &mut self.buf_b,
            );

            // Layer 2: 32x32x8 -> 16x16x16
            ea_conv3x3_relu_pool(
                &self.buf_b, 32, 32, CONV2_IN,
                &self.conv2_weights_relaid, model.conv2_bias(),
                CONV2_OUT, &mut self.buf_a,
            );

            // Layer 3: 16x16x16 -> 8x8x32
            ea_conv3x3_relu_pool(
                &self.buf_a, 16, 16, CONV3_IN,
                &self.conv3_weights_relaid, model.conv3_bias(),
                CONV3_OUT, &mut self.buf_b,
            );

            // GAP + FC + softmax
            let mut scores = [0.0f32; NUM_CLASSES];
            ea_gap_fc_softmax(
                &self.buf_b, 8, 8, CONV3_OUT,
                model.fc_weights(), model.fc_bias(),
                &mut scores, NUM_CLASSES,
            );

            // Argmax
            let mut best_id = 0;
            let mut best_score = scores[0];
            for i in 1..NUM_CLASSES {
                if scores[i] > best_score {
                    best_score = scores[i];
                    best_id = i;
                }
            }

            Classification {
                class_id: best_id,
                class_name: CLASS_NAMES[best_id],
                confidence: best_score,
                scores,
            }
        }

        #[cfg(not(feature = "ea"))]
        {
            // Step 2: Layer 1 — 64x64x1 -> 32x32x8
            fused_conv_relu_pool(
                &self.buf_a, 64, 64, CONV1_IN,
                model.conv1_weights(), model.conv1_bias(),
                CONV1_OUT, &mut self.buf_b,
            );

            // Step 3: Layer 2 — 32x32x8 -> 16x16x16
            fused_conv_relu_pool(
                &self.buf_b, 32, 32, CONV2_IN,
                model.conv2_weights(), model.conv2_bias(),
                CONV2_OUT, &mut self.buf_a,
            );

            // Step 4: Layer 3 — 16x16x16 -> 8x8x32
            fused_conv_relu_pool(
                &self.buf_a, 16, 16, CONV3_IN,
                model.conv3_weights(), model.conv3_bias(),
                CONV3_OUT, &mut self.buf_b,
            );

            // Step 5: Fused GAP + FC + softmax
            let scores = fused_gap_fc_softmax(
                &self.buf_b, 8, 8, CONV3_OUT,
                model.fc_weights(), model.fc_bias(),
            );

            // Step 6: Argmax
            let mut best_id = 0;
            let mut best_score = scores[0];
            for i in 1..NUM_CLASSES {
                if scores[i] > best_score {
                    best_score = scores[i];
                    best_id = i;
                }
            }

            Classification {
                class_id: best_id,
                class_name: CLASS_NAMES[best_id],
                confidence: best_score,
                scores,
            }
        }
    }
}

/// Compute conv3x3 + bias at a single spatial position, applying zero-padding.
/// Returns the pre-activation value (before ReLU). Stays in a register.
#[inline(always)]
fn conv3x3_at(
    input: &[f32], in_w: usize, in_h: usize, c_in: usize,
    weights: &[i8], bias_val: f32,
    oc: usize, ox: usize, oy: usize,
) -> f32 {
    let mut acc = bias_val;
    for ic in 0..c_in {
        let w_base = (oc * c_in + ic) * 9;
        for ky in 0..3usize {
            for kx in 0..3usize {
                let iy = oy as isize + ky as isize - 1;
                let ix = ox as isize + kx as isize - 1;
                if iy >= 0 && iy < in_h as isize && ix >= 0 && ix < in_w as isize {
                    let pixel = input[(iy as usize * in_w + ix as usize) * c_in + ic];
                    let w = weights[w_base + ky * 3 + kx] as f32;
                    acc += pixel * w;
                }
            }
        }
    }
    acc
}

/// Fused conv3x3 + ReLU + 2x2 max-pool.
///
/// For each output channel and each 2x2 pool window, we compute conv+bias at
/// all 4 positions, apply ReLU, take max -- the intermediate conv/relu values
/// never leave registers. Only the final pooled value is written to output.
///
/// Input layout: HWC (height, width, channels) -- row-major with channels innermost.
/// Output layout: HWC at (in_h/2, in_w/2, c_out).
fn fused_conv_relu_pool(
    input: &[f32], in_w: usize, in_h: usize, c_in: usize,
    weights: &[i8], bias: &[i8],
    c_out: usize, output: &mut [f32],
) {
    let out_w = in_w / 2;
    let out_h = in_h / 2;

    for oc in 0..c_out {
        let bias_val = bias[oc] as f32;
        for py in 0..out_h {
            for px in 0..out_w {
                // The 2x2 pool window covers input positions:
                // (px*2, py*2), (px*2+1, py*2), (px*2, py*2+1), (px*2+1, py*2+1)
                let mut pool_max = f32::NEG_INFINITY;
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let ix = px * 2 + dx;
                        let iy = py * 2 + dy;
                        // Conv3x3 + bias (stays in register)
                        let val = conv3x3_at(
                            input, in_w, in_h, c_in,
                            weights, bias_val, oc, ix, iy,
                        );
                        // ReLU (stays in register)
                        let val = if val > 0.0 { val } else { 0.0 };
                        // Max pool accumulation (stays in register)
                        if val > pool_max {
                            pool_max = val;
                        }
                    }
                }
                // ONE write per pool window
                output[(py * out_w + px) * c_out + oc] = pool_max;
            }
        }
    }
}

/// Fused global average pooling + fully-connected + softmax.
///
/// For each input channel, we sum all spatial values (GAP), divide by w*h,
/// and immediately accumulate into all FC outputs. The per-channel GAP
/// average stays in a register -- never written to an intermediate buffer.
fn fused_gap_fc_softmax(
    input: &[f32], w: usize, h: usize, channels: usize,
    fc_weights: &[i8], fc_bias: &[i8],
) -> [f32; NUM_CLASSES] {
    let mut scores = [0.0f32; NUM_CLASSES];

    // Initialize from FC bias
    for c in 0..NUM_CLASSES {
        scores[c] = fc_bias[c] as f32;
    }

    let spatial = (w * h) as f32;

    // For each channel: compute GAP (sum/spatial), then immediately
    // multiply into all FC outputs. The channel average stays in register.
    for ch in 0..channels {
        let mut sum = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                sum += input[(y * w + x) * channels + ch];
            }
        }
        let avg = sum / spatial; // stays in register

        // Accumulate into all class scores
        for c in 0..NUM_CLASSES {
            scores[c] += avg * fc_weights[ch * NUM_CLASSES + c] as f32;
        }
    }

    // Softmax
    softmax(&mut scores);
    scores
}

/// Numerically stable softmax in-place.
fn softmax(scores: &mut [f32; NUM_CLASSES]) {
    let mut max_val = scores[0];
    for i in 1..NUM_CLASSES {
        if scores[i] > max_val {
            max_val = scores[i];
        }
    }
    let mut sum = 0.0f32;
    for i in 0..NUM_CLASSES {
        scores[i] = (scores[i] - max_val).exp();
        sum += scores[i];
    }
    for i in 0..NUM_CLASSES {
        scores[i] /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_dummy_model() {
        let model = Model::dummy();
        let frame = Frame::new(INPUT_SIZE, INPUT_SIZE);
        let mut engine = InferenceEngine::new();
        let result = engine.classify(&frame, &model);
        // Zero weights + zero bias means all conv outputs are 0, GAP is 0,
        // FC outputs are all 0 -> softmax gives uniform 1/NUM_CLASSES
        let expected = 1.0 / NUM_CLASSES as f32;
        for &s in result.scores.iter() {
            assert!((s - expected).abs() < 1e-5, "expected ~{}, got {}", expected, s);
        }
    }

    #[test]
    fn test_classify_nonzero_frame() {
        let model = Model::dummy();
        let mut frame = Frame::new(INPUT_SIZE, INPUT_SIZE);
        // Fill with a diagonal pattern
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                frame.set_pixel(x, y, ((x + y) % 256) as u8);
            }
        }
        let mut engine = InferenceEngine::new();
        let result = engine.classify(&frame, &model);
        // Even with a patterned frame, zero-weight model still gives uniform scores
        let sum: f32 = result.scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probabilities must sum to 1.0, got {}",
            sum
        );
        for &s in result.scores.iter() {
            assert!(s >= 0.0 && s <= 1.0, "probability out of range: {}", s);
        }
    }

    #[test]
    fn test_softmax() {
        let mut scores = [1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut scores);
        // Must be monotonically increasing
        for i in 1..NUM_CLASSES {
            assert!(
                scores[i] > scores[i - 1],
                "softmax should be monotonically increasing: {:?}",
                scores,
            );
        }
        // Must sum to 1.0
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_classification_has_valid_class() {
        let model = Model::dummy();
        let frame = Frame::new(INPUT_SIZE, INPUT_SIZE);
        let mut engine = InferenceEngine::new();
        let result = engine.classify(&frame, &model);
        assert!(result.class_id < NUM_CLASSES, "class_id out of range");
        assert!(
            CLASS_NAMES.contains(&result.class_name),
            "class_name not in CLASS_NAMES",
        );
        assert!(
            result.confidence > 0.0 && result.confidence <= 1.0,
            "confidence {} not in (0, 1]",
            result.confidence,
        );
    }

    #[test]
    fn test_max_pool_2x2_via_fused() {
        // 4x4 input with 1 channel, manually set values so we can verify pooling.
        // After conv3x3 with weight=1 for center pixel (identity-like) + relu + pool,
        // the pool should pick the max from each 2x2 window.
        let c_in = 1;
        let c_out = 1;
        let in_w = 4;
        let in_h = 4;
        // Input: 4x4x1 with known values
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        // Weights: 3x3 kernel, identity-like (only center=1, rest=0)
        // Layout: [c_out][c_in][3][3] = [1][1][3][3]
        let weights: Vec<i8> = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        let bias: Vec<i8> = vec![0];

        let mut output = vec![0.0f32; 2 * 2 * c_out];
        fused_conv_relu_pool(&input, in_w, in_h, c_in, &weights, &bias, c_out, &mut output);

        // With identity kernel (center=1), conv output equals input.
        // ReLU is no-op (all positive). Max pool 2x2:
        //   top-left: max(1,2,5,6) = 6
        //   top-right: max(3,4,7,8) = 8
        //   bottom-left: max(9,10,13,14) = 14
        //   bottom-right: max(11,12,15,16) = 16
        assert!((output[0] - 6.0).abs() < 1e-5, "top-left pool: {}", output[0]);
        assert!((output[1] - 8.0).abs() < 1e-5, "top-right pool: {}", output[1]);
        assert!((output[2] - 14.0).abs() < 1e-5, "bottom-left pool: {}", output[2]);
        assert!((output[3] - 16.0).abs() < 1e-5, "bottom-right pool: {}", output[3]);
    }

    #[test]
    fn test_global_avg_pool_via_fused() {
        // 4x4 input with 2 channels.
        // Channel 0: all 2.0, Channel 1: all 4.0
        // GAP should produce [2.0, 4.0].
        // FC weights: identity-like mapping 2 channels -> 4 classes:
        //   class 0 gets ch0*1 + ch1*0 = 2.0
        //   class 1 gets ch0*0 + ch1*1 = 4.0
        //   class 2, 3 get 0
        let w = 4;
        let h = 4;
        let channels = 2;
        let mut input = vec![0.0f32; w * h * channels];
        for y in 0..h {
            for x in 0..w {
                input[(y * w + x) * channels + 0] = 2.0;
                input[(y * w + x) * channels + 1] = 4.0;
            }
        }

        // FC weights: [channels * NUM_CLASSES] = [2 * 4]
        // ch0 -> [1, 0, 0, 0], ch1 -> [0, 1, 0, 0]
        let fc_weights: Vec<i8> = vec![
            1, 0, 0, 0, // ch0 -> classes
            0, 1, 0, 0, // ch1 -> classes
        ];
        let fc_bias: Vec<i8> = vec![0, 0, 0, 0];

        let scores = fused_gap_fc_softmax(&input, w, h, channels, &fc_weights, &fc_bias);

        // Before softmax, raw scores would be [2.0, 4.0, 0.0, 0.0].
        // After softmax, class 1 (score 4.0) should have highest probability.
        assert!(scores[1] > scores[0], "class 1 should beat class 0");
        assert!(scores[0] > scores[2], "class 0 should beat class 2");
        assert!((scores[2] - scores[3]).abs() < 1e-5, "class 2 and 3 should be equal");

        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1.0, got {}", sum);
    }
}
