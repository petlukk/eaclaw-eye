use std::time::{SystemTime, UNIX_EPOCH};

use super::frame::Frame;

pub trait Camera: Send {
    fn grab(&mut self, frame: &mut Frame) -> Result<bool, String>;
    fn resolution(&self) -> (u32, u32);
}

/// Generates frames with a moving 20x20 bright spot for testing.
pub struct MockCamera {
    width: u32,
    height: u32,
    frame_count: u32,
}

impl MockCamera {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frame_count: 0,
        }
    }
}

impl Camera for MockCamera {
    fn grab(&mut self, frame: &mut Frame) -> Result<bool, String> {
        if frame.width != self.width || frame.height != self.height {
            return Err(format!(
                "Frame size {}x{} does not match camera {}x{}",
                frame.width, frame.height, self.width, self.height
            ));
        }

        // Clear frame to 0
        for byte in frame.data.iter_mut() {
            *byte = 0;
        }

        // Draw 20x20 bright spot at moving position
        let spot_x = (self.frame_count * 5) % self.width;
        let spot_y = self.height / 2;
        for dy in 0..20u32 {
            for dx in 0..20u32 {
                let px = spot_x + dx;
                let py = spot_y + dy;
                if px < self.width && py < self.height {
                    frame.set_pixel(px, py, 255);
                }
            }
        }

        // Set timestamp
        frame.timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

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
        let mut cam = MockCamera::new(320, 240);
        let mut frame = Frame::new(320, 240);
        let result = cam.grab(&mut frame);
        assert!(result.is_ok());
        assert!(result.unwrap());
        assert!(frame.timestamp_ms > 0);
        assert!(frame.data.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_mock_camera_spot_moves() {
        let mut cam = MockCamera::new(320, 240);
        let mut frame1 = Frame::new(320, 240);
        let mut frame2 = Frame::new(320, 240);
        cam.grab(&mut frame1).unwrap();
        cam.grab(&mut frame2).unwrap();
        assert_ne!(frame1.data, frame2.data);
    }

    #[test]
    fn test_mock_camera_size_mismatch() {
        let mut cam = MockCamera::new(320, 240);
        let mut frame = Frame::new(640, 480);
        let result = cam.grab(&mut frame);
        assert!(result.is_err());
    }
}
