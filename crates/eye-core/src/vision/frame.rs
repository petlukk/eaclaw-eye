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
        let frame = Frame::new(64, 64);
        assert_eq!(frame.len(), 64 * 64);
        assert_eq!(frame.pixel(0, 0), 0);
    }

    #[test]
    fn test_frame_set_pixel() {
        let mut frame = Frame::new(64, 64);
        frame.set_pixel(5, 3, 200);
        assert_eq!(frame.pixel(5, 3), 200);
    }

    #[test]
    fn test_crop_and_scale() {
        let mut frame = Frame::new(100, 100);
        // Fill a 20x20 block at (10,10) with 255
        for y in 10..30 {
            for x in 10..30 {
                frame.set_pixel(x, y, 255);
            }
        }
        let cropped = frame.crop_and_scale(10, 10, 20, 20, 8);
        assert_eq!(cropped.width, 8);
        assert_eq!(cropped.height, 8);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(cropped.pixel(x, y), 255);
            }
        }
    }

    #[test]
    fn test_crop_and_scale_downscale() {
        let frame = Frame::new(640, 480);
        let cropped = frame.crop_and_scale(0, 0, 640, 480, 64);
        assert_eq!(cropped.width, 64);
        assert_eq!(cropped.height, 64);
    }
}
