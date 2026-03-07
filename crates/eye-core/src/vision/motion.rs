use super::frame::Frame;

#[derive(Debug, Clone, PartialEq)]
pub struct MotionBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub pixel_count: u32,
}

pub struct MotionDetector {
    pub threshold: u8,
    pub min_pixels: u32,
    cell_size: u32,
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

    pub fn detect(&mut self, prev: &Frame, curr: &Frame) -> Vec<MotionBox> {
        assert_eq!(prev.width, curr.width);
        assert_eq!(prev.height, curr.height);

        let width = prev.width;
        let height = prev.height;
        let cols = (width + self.cell_size - 1) / self.cell_size;
        let rows = (height + self.cell_size - 1) / self.cell_size;

        self.grid_counts.resize((cols * rows) as usize, 0);
        self.grid_counts.fill(0);

        let mut total_changed = 0u32;

        // Single pass: diff + threshold + grid accumulate (fused)
        // diff value stays in register -- never written to memory
        for y in 0..height {
            let gy = y / self.cell_size;
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let diff =
                    (prev.data[idx] as i16 - curr.data[idx] as i16).unsigned_abs() as u8;
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

        // Extract boxes from grid cells that exceed the threshold
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

fn boxes_overlap(a: &MotionBox, b: &MotionBox, gap: u32) -> bool {
    let a_right = a.x + a.w;
    let b_right = b.x + b.w;
    let a_bottom = a.y + a.h;
    let b_bottom = b.y + b.h;

    // Check if boxes are within `gap` of each other on both axes
    let x_close = a.x <= b_right + gap && b.x <= a_right + gap;
    let y_close = a.y <= b_bottom + gap && b.y <= a_bottom + gap;

    x_close && y_close
}

fn merge_boxes(boxes: &mut Vec<MotionBox>, gap: u32) {
    loop {
        let mut merged = false;
        let mut i = 0;
        while i < boxes.len() {
            let mut j = i + 1;
            while j < boxes.len() {
                if boxes_overlap(&boxes[i], &boxes[j], gap) {
                    let b = boxes.remove(j);
                    let a = &mut boxes[i];

                    let new_x = a.x.min(b.x);
                    let new_y = a.y.min(b.y);
                    let new_right = (a.x + a.w).max(b.x + b.w);
                    let new_bottom = (a.y + a.h).max(b.y + b.h);

                    a.x = new_x;
                    a.y = new_y;
                    a.w = new_right - new_x;
                    a.h = new_bottom - new_y;
                    a.pixel_count += b.pixel_count;

                    merged = true;
                    // Don't increment j; re-check from the same position
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if !merged {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_motion() {
        let mut detector = MotionDetector::new(10, 5);
        let frame_a = Frame::new(64, 64);
        let frame_b = Frame::new(64, 64);
        let boxes = detector.detect(&frame_a, &frame_b);
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_motion_detected() {
        let mut detector = MotionDetector::new(10, 5);
        let prev = Frame::new(128, 128);
        let mut curr = Frame::new(128, 128);

        // Paint a 20x20 bright block at (10,10)
        for y in 10..30 {
            for x in 10..30 {
                curr.set_pixel(x, y, 200);
            }
        }

        let boxes = detector.detect(&prev, &curr);
        assert!(!boxes.is_empty(), "expected motion boxes");

        // The motion boxes should cover the bright block region
        for b in &boxes {
            assert!(b.pixel_count > 0);
        }

        // Verify every changed pixel is inside at least one box
        for y in 10u32..30 {
            for x in 10u32..30 {
                let covered = boxes
                    .iter()
                    .any(|b| x >= b.x && x < b.x + b.w && y >= b.y && y < b.y + b.h);
                assert!(covered, "pixel ({x},{y}) not covered by any box");
            }
        }
    }

    #[test]
    fn test_below_threshold_ignored() {
        let mut detector = MotionDetector::new(100, 5);
        let prev = Frame::new(64, 64);
        let mut curr = Frame::new(64, 64);

        // Change all pixels by 50, but threshold is 100
        for px in curr.data.iter_mut() {
            *px = 50;
        }

        let boxes = detector.detect(&prev, &curr);
        assert!(boxes.is_empty(), "changes below threshold should be ignored");
    }

    #[test]
    fn test_min_pixels_filter() {
        let mut detector = MotionDetector::new(10, 100);
        let prev = Frame::new(64, 64);
        let mut curr = Frame::new(64, 64);

        // Only change 4 pixels
        curr.set_pixel(0, 0, 200);
        curr.set_pixel(1, 0, 200);
        curr.set_pixel(2, 0, 200);
        curr.set_pixel(3, 0, 200);

        let boxes = detector.detect(&prev, &curr);
        assert!(boxes.is_empty(), "too few changed pixels, should produce no boxes");
    }

    #[test]
    fn test_merge_adjacent_boxes() {
        let mut detector = MotionDetector::new(10, 5);
        let prev = Frame::new(128, 128);
        let mut curr = Frame::new(128, 128);

        // Create two adjacent bright regions that span neighboring grid cells.
        // Cell size is 32, so cells [0,0] and [1,0] are adjacent.
        // Fill a block in cell (0,0): x 0..20, y 0..32
        for y in 0..32 {
            for x in 0..20 {
                curr.set_pixel(x, y, 200);
            }
        }
        // Fill a block in cell (1,0): x 32..52, y 0..32
        for y in 0..32 {
            for x in 32..52 {
                curr.set_pixel(x, y, 200);
            }
        }

        let boxes = detector.detect(&prev, &curr);

        // Adjacent cells should merge into one box
        assert_eq!(
            boxes.len(),
            1,
            "two adjacent regions should merge into one box, got {boxes:?}"
        );

        let b = &boxes[0];
        // Merged box should span from x=0 to at least x=64 (two 32-wide cells)
        assert_eq!(b.x, 0);
        assert!(b.w >= 64, "merged width should cover both cells, got {}", b.w);
    }
}
