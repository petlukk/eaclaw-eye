use super::capture::Camera;
use super::frame::Frame;
use super::inference::InferenceEngine;
use super::model::{Classification, Model, INPUT_SIZE};
use super::motion::{MotionBox, MotionDetector};

#[derive(Debug, Clone)]
pub struct Detection {
    pub classification: Classification,
    pub region: MotionBox,
    pub timestamp_ms: u64,
}

impl Detection {
    /// Human-readable alert line, e.g. "Person detected (92%) — 22:14:03"
    pub fn alert_text(&self) -> String {
        let pct = (self.classification.confidence * 100.0) as u32;
        let total_secs = (self.timestamp_ms / 1000) as u32;
        let hh = (total_secs / 3600) % 24;
        let mm = (total_secs % 3600) / 60;
        let ss = total_secs % 60;
        let name = capitalize(self.classification.class_name);
        format!("{name} detected ({pct}%) \u{2014} {hh:02}:{mm:02}:{ss:02}")
    }

    /// Machine-readable recall line, e.g. "person confidence=0.92 at=1234567"
    pub fn recall_text(&self) -> String {
        format!(
            "{} confidence={:.2} at={}",
            self.classification.class_name, self.classification.confidence, self.timestamp_ms
        )
    }
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

pub struct Pipeline {
    motion: MotionDetector,
    engine: InferenceEngine,
    model: Model,
    prev_frame: Frame,
    curr_frame: Frame,
    patch: Frame,
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

    /// Run the two-stage detection pipeline on the next camera frame.
    ///
    /// Stage 1: motion detection (prev vs curr) rejects ~95% of frames.
    /// Stage 2: for each motion region, crop+scale to 64x64 and classify.
    pub fn process(&mut self, camera: &mut dyn Camera) -> Result<Vec<Detection>, String> {
        // Swap prev/curr so the old curr becomes the new prev
        std::mem::swap(&mut self.prev_frame, &mut self.curr_frame);

        // Grab a new frame into curr
        camera.grab(&mut self.curr_frame)?;

        // Stage 1: motion detection
        let boxes = self.motion.detect(&self.prev_frame, &self.curr_frame);
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 2: classify each motion region
        let timestamp = self.curr_frame.timestamp_ms;
        let mut detections = Vec::new();

        for region in boxes {
            // Crop the motion region and scale to CNN input size
            self.patch = self.curr_frame.crop_and_scale(
                region.x,
                region.y,
                region.w,
                region.h,
                INPUT_SIZE,
            );

            let classification = self.engine.classify(&self.patch, &self.model);

            // Filter: skip "nothing" (class 0) and low confidence
            if classification.class_id == 0 {
                continue;
            }
            if classification.confidence < self.min_confidence {
                continue;
            }

            detections.push(Detection {
                classification,
                region,
                timestamp_ms: timestamp,
            });
        }

        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::capture::MockCamera;

    #[test]
    fn test_pipeline_no_motion_first_frame() {
        // First frame: prev and curr are both zeroed. With a dummy model
        // (zero weights), softmax gives uniform 1/NUM_CLASSES = 0.25 for each
        // class, which is below min_confidence=0.5. Even if there were motion,
        // nothing would pass the confidence filter.
        let model = Model::dummy();
        let mut pipeline = Pipeline::new(model, 320, 240);
        let mut cam = MockCamera::new(320, 240);
        let result = pipeline.process(&mut cam).unwrap();
        assert!(
            result.is_empty(),
            "first frame should produce no detections (confidence 0.25 < 0.5)"
        );
    }

    #[test]
    fn test_pipeline_processes_frames() {
        let model = Model::dummy();
        let mut pipeline = Pipeline::new(model, 320, 240);
        let mut cam = MockCamera::new(320, 240);

        // Frame 1: establishes baseline
        let r1 = pipeline.process(&mut cam);
        assert!(r1.is_ok());

        // Frame 2: MockCamera moves the bright spot, so motion is possible
        let r2 = pipeline.process(&mut cam);
        assert!(r2.is_ok());
    }

    #[test]
    fn test_detection_alert_text() {
        let det = Detection {
            classification: Classification {
                class_id: 1,
                class_name: "person",
                confidence: 0.92,
                scores: [0.02, 0.92, 0.03, 0.03],
            },
            region: MotionBox {
                x: 10,
                y: 20,
                w: 30,
                h: 40,
                pixel_count: 100,
            },
            // 22:14:03 = 22*3600 + 14*60 + 3 = 79200 + 840 + 3 = 80043 seconds
            timestamp_ms: 80043 * 1000,
        };

        let text = det.alert_text();
        assert_eq!(text, "Person detected (92%) \u{2014} 22:14:03");
    }

    #[test]
    fn test_detection_recall_text() {
        let det = Detection {
            classification: Classification {
                class_id: 1,
                class_name: "person",
                confidence: 0.92,
                scores: [0.02, 0.92, 0.03, 0.03],
            },
            region: MotionBox {
                x: 10,
                y: 20,
                w: 30,
                h: 40,
                pixel_count: 100,
            },
            timestamp_ms: 1234567,
        };

        let text = det.recall_text();
        assert_eq!(text, "person confidence=0.92 at=1234567");
    }
}
