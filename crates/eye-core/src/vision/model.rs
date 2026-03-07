/// CNN model weight format for L1-resident inference.
///
/// 3-layer conv net: 1->8->16->32 channels, 3x3 kernels, followed by
/// global average pool and a 32->4 fully-connected classifier.
/// Total: 6020 int8 parameters (~6 KB, fits comfortably in Pi 5 L1d).

pub const NUM_CLASSES: usize = 4;
pub const INPUT_SIZE: u32 = 64;
pub const CLASS_NAMES: [&str; NUM_CLASSES] = ["nothing", "person", "vehicle", "animal"];

pub const CONV1_IN: usize = 1;
pub const CONV1_OUT: usize = 8;
pub const CONV2_IN: usize = 8;
pub const CONV2_OUT: usize = 16;
pub const CONV3_IN: usize = 16;
pub const CONV3_OUT: usize = 32;
pub const FC_IN: usize = 32;

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

#[derive(Debug, Clone)]
pub struct Classification {
    pub class_id: usize,
    pub class_name: &'static str,
    pub confidence: f32,
    pub scores: [f32; NUM_CLASSES],
}

pub struct Model {
    pub data: Vec<i8>,
}

// Byte offsets for each layer slice within `data`.
const OFF_CONV1_W: usize = 0;
const OFF_CONV1_B: usize = OFF_CONV1_W + CONV1_WEIGHTS;
const OFF_CONV2_W: usize = OFF_CONV1_B + CONV1_BIAS;
const OFF_CONV2_B: usize = OFF_CONV2_W + CONV2_WEIGHTS;
const OFF_CONV3_W: usize = OFF_CONV2_B + CONV2_BIAS;
const OFF_CONV3_B: usize = OFF_CONV3_W + CONV3_WEIGHTS;
const OFF_FC_W: usize = OFF_CONV3_B + CONV3_BIAS;
const OFF_FC_B: usize = OFF_FC_W + FC_WEIGHTS;

impl Model {
    /// Load model from raw int8 weight bytes.
    /// The byte slice is reinterpreted as `i8`; length must equal `TOTAL_PARAMS`.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != TOTAL_PARAMS {
            return Err(format!(
                "expected {} bytes, got {}",
                TOTAL_PARAMS,
                data.len()
            ));
        }
        // Safe: u8 and i8 have the same size/alignment; we copy into a new Vec.
        let signed: Vec<i8> = data.iter().map(|&b| b as i8).collect();
        Ok(Self { data: signed })
    }

    /// Create a zero-weight model for testing.
    pub fn dummy() -> Self {
        Self {
            data: vec![0i8; TOTAL_PARAMS],
        }
    }

    pub fn conv1_weights(&self) -> &[i8] {
        &self.data[OFF_CONV1_W..OFF_CONV1_W + CONV1_WEIGHTS]
    }

    pub fn conv1_bias(&self) -> &[i8] {
        &self.data[OFF_CONV1_B..OFF_CONV1_B + CONV1_BIAS]
    }

    pub fn conv2_weights(&self) -> &[i8] {
        &self.data[OFF_CONV2_W..OFF_CONV2_W + CONV2_WEIGHTS]
    }

    pub fn conv2_bias(&self) -> &[i8] {
        &self.data[OFF_CONV2_B..OFF_CONV2_B + CONV2_BIAS]
    }

    pub fn conv3_weights(&self) -> &[i8] {
        &self.data[OFF_CONV3_W..OFF_CONV3_W + CONV3_WEIGHTS]
    }

    pub fn conv3_bias(&self) -> &[i8] {
        &self.data[OFF_CONV3_B..OFF_CONV3_B + CONV3_BIAS]
    }

    pub fn fc_weights(&self) -> &[i8] {
        &self.data[OFF_FC_W..OFF_FC_W + FC_WEIGHTS]
    }

    pub fn fc_bias(&self) -> &[i8] {
        &self.data[OFF_FC_B..OFF_FC_B + FC_BIAS]
    }

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
        // Pi 5 L1d cache is 64 KB.
        assert!(TOTAL_PARAMS < 65536);
    }

    #[test]
    fn test_dummy_model() {
        let m = Model::dummy();
        assert_eq!(m.size_bytes(), TOTAL_PARAMS);
        assert_eq!(m.conv1_weights().len(), CONV1_WEIGHTS);
        assert_eq!(m.conv1_bias().len(), CONV1_BIAS);
        assert_eq!(m.conv2_weights().len(), CONV2_WEIGHTS);
        assert_eq!(m.conv2_bias().len(), CONV2_BIAS);
        assert_eq!(m.conv3_weights().len(), CONV3_WEIGHTS);
        assert_eq!(m.conv3_bias().len(), CONV3_BIAS);
        assert_eq!(m.fc_weights().len(), FC_WEIGHTS);
        assert_eq!(m.fc_bias().len(), FC_BIAS);
    }

    #[test]
    fn test_from_bytes_valid() {
        let data = vec![0u8; TOTAL_PARAMS];
        let m = Model::from_bytes(&data);
        assert!(m.is_ok());
        assert_eq!(m.unwrap().size_bytes(), TOTAL_PARAMS);
    }

    #[test]
    fn test_from_bytes_wrong_size() {
        let data = vec![0u8; 100];
        let m = Model::from_bytes(&data);
        assert!(m.is_err());
    }

    #[test]
    fn test_layer_slices_contiguous() {
        let m = Model::dummy();
        let total = m.conv1_weights().len()
            + m.conv1_bias().len()
            + m.conv2_weights().len()
            + m.conv2_bias().len()
            + m.conv3_weights().len()
            + m.conv3_bias().len()
            + m.fc_weights().len()
            + m.fc_bias().len();
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
