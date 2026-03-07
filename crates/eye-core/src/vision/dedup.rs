//! Detection deduplication using VectorStore recall.
//!
//! Uses byte-histogram cosine similarity to suppress repeated detections
//! of the same class in a short time window. Each detection's `recall_text()`
//! is indexed; if a new detection is too similar to a recent one, it is
//! flagged as a duplicate.

use eaclaw_core::recall::VectorStore;

use super::pipeline::Detection;

/// Deduplicates detections using VectorStore similarity search.
///
/// Every detection is indexed (even duplicates) so the ring buffer
/// tracks recency accurately. A detection is duplicate when its
/// `recall_text()` cosine similarity to the top existing entry
/// exceeds `similarity_threshold`.
pub struct DetectionDedup {
    store: VectorStore,
    pub similarity_threshold: f32,
}

impl DetectionDedup {
    /// Create a new dedup store with the given ring-buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            store: VectorStore::with_capacity(capacity),
            similarity_threshold: 0.90,
        }
    }

    /// Returns `true` if this detection is a duplicate of a recent one.
    ///
    /// Always indexes the detection for future comparisons, even when
    /// it is a duplicate. This keeps the recency window accurate.
    pub fn is_duplicate(&mut self, detection: &Detection) -> bool {
        let text = Self::dedup_key(detection);

        let duplicate = if self.store.is_empty() {
            false
        } else {
            let results = self.store.recall(&text, 1);
            match results.first() {
                Some(r) => r.score > self.similarity_threshold,
                None => false,
            }
        };

        self.store.insert(&text);
        duplicate
    }

    /// Build a dedup key that emphasizes the class name.
    ///
    /// The raw `recall_text()` shares too many characters across classes
    /// (e.g. "confidence=0.90 at=1000") which inflates byte-histogram
    /// cosine similarity. Repeating the class name amplifies its weight
    /// in the histogram so different classes stay below threshold.
    fn dedup_key(detection: &Detection) -> String {
        let name = detection.classification.class_name;
        format!(
            "{name} {name} {name} {name} at={}",
            detection.timestamp_ms
        )
    }

    /// Number of indexed detections (capped at capacity).
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if no detections have been indexed.
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
    use crate::vision::model::{Classification, NUM_CLASSES};
    use crate::vision::motion::MotionBox;

    fn make_detection(class: &'static str, class_id: usize, ts: u64) -> Detection {
        Detection {
            classification: Classification {
                class_id,
                class_name: class,
                confidence: 0.9,
                scores: [0.0; NUM_CLASSES],
            },
            bbox: MotionBox {
                x: 0,
                y: 0,
                w: 50,
                h: 50,
                pixel_count: 100,
            },
            timestamp_ms: ts,
        }
    }

    #[test]
    fn test_first_detection_not_duplicate() {
        let mut dedup = DetectionDedup::new(64);
        let det = make_detection("person", 1, 1000);
        assert!(!dedup.is_duplicate(&det));
    }

    #[test]
    fn test_same_detection_is_duplicate() {
        let mut dedup = DetectionDedup::new(64);
        let det1 = make_detection("person", 1, 1000);
        let det2 = make_detection("person", 1, 1001);
        assert!(!dedup.is_duplicate(&det1));
        assert!(dedup.is_duplicate(&det2));
    }

    #[test]
    fn test_different_class_not_duplicate() {
        let mut dedup = DetectionDedup::new(64);
        let person = make_detection("person", 1, 1000);
        let vehicle = make_detection("vehicle", 2, 1001);
        assert!(!dedup.is_duplicate(&person));
        assert!(!dedup.is_duplicate(&vehicle));
    }

    #[test]
    fn test_clear() {
        let mut dedup = DetectionDedup::new(64);
        let det = make_detection("person", 1, 1000);
        dedup.is_duplicate(&det);
        assert_eq!(dedup.len(), 1);
        dedup.clear();
        assert_eq!(dedup.len(), 0);
    }
}
