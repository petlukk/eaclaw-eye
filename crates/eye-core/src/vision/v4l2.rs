//! V4L2 camera capture for Raspberry Pi Camera Module v2.
//! Feature-gated behind `v4l2` since it requires Linux + camera hardware.

use super::capture::Camera;
use super::frame::Frame;

use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::Device;
use v4l::FourCC;

pub struct V4l2Camera {
    stream: Stream,
    width: u32,
    height: u32,
}

impl V4l2Camera {
    pub fn open(device_path: &str, width: u32, height: u32) -> Result<Self, String> {
        let dev = Device::with_path(device_path)
            .map_err(|e| format!("failed to open {device_path}: {e}"))?;

        let mut fmt = dev.format().map_err(|e| format!("get format: {e}"))?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = FourCC::new(b"GREY");
        let fmt = dev
            .set_format(&fmt)
            .map_err(|e| format!("set format: {e}"))?;

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

impl Camera for V4l2Camera {
    fn grab(&mut self, frame: &mut Frame) -> Result<bool, String> {
        let (buf, _meta) = self.stream.next().map_err(|e| format!("capture: {e}"))?;

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
