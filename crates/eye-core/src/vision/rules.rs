use super::model::{CLASS_NAMES, NUM_CLASSES};
use super::pipeline::Detection;
use serde::{Deserialize, Serialize};
use serde_json;

/// Case-insensitive lookup of a class name in CLASS_NAMES.
fn class_index(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    CLASS_NAMES.iter().position(|&n| n == lower)
}

/// Rules engine that decides whether a detection should trigger an alert.
///
/// Filters on: class enablement, minimum confidence, mute window, and cooldown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRules {
    pub class_enabled: [bool; NUM_CLASSES],
    pub min_confidence: f32,
    pub mute_until_ms: u64,
    pub cooldown_ms: u64,
    #[serde(skip)]
    pub last_alert_ms: u64,
}

impl Default for AlertRules {
    fn default() -> Self {
        Self {
            class_enabled: [false, true, true, false],
            min_confidence: 0.5,
            mute_until_ms: 0,
            cooldown_ms: 30_000,
            last_alert_ms: 0,
        }
    }
}

impl AlertRules {
    /// Returns true if the given detection should produce an alert right now.
    ///
    /// Checks (in order):
    /// 1. Class must be enabled.
    /// 2. Confidence must meet the minimum threshold.
    /// 3. Current time must be past the mute window.
    /// 4. Enough time must have elapsed since the last alert (cooldown).
    pub fn should_alert(&self, detection: &Detection, now_ms: u64) -> bool {
        let class_id = detection.classification.class_id;

        // Class must be valid and enabled
        if class_id >= NUM_CLASSES || !self.class_enabled[class_id] {
            return false;
        }

        // Confidence check
        if detection.classification.confidence < self.min_confidence {
            return false;
        }

        // Mute check
        if now_ms < self.mute_until_ms {
            return false;
        }

        // Cooldown check
        if self.last_alert_ms > 0 && now_ms < self.last_alert_ms + self.cooldown_ms {
            return false;
        }

        true
    }

    /// Record that an alert was fired at the given timestamp.
    pub fn record_alert(&mut self, now_ms: u64) {
        self.last_alert_ms = now_ms;
    }

    /// Enable alerting for a class by name (case-insensitive).
    /// Returns false if the name is not a valid class.
    pub fn enable_class(&mut self, name: &str) -> bool {
        match class_index(name) {
            Some(idx) => {
                self.class_enabled[idx] = true;
                true
            }
            None => false,
        }
    }

    /// Disable alerting for a class by name (case-insensitive).
    /// Returns false if the name is not a valid class.
    pub fn disable_class(&mut self, name: &str) -> bool {
        match class_index(name) {
            Some(idx) => {
                self.class_enabled[idx] = false;
                true
            }
            None => false,
        }
    }

    /// Validate that all fields are within acceptable ranges.
    ///
    /// Guards external input (e.g. JSON config files, API calls).
    /// Internal construction via `Default` is always valid.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_confidence <= 0.0 || self.min_confidence > 1.0 {
            return Err(format!(
                "min_confidence must be in (0.0, 1.0], got {}",
                self.min_confidence
            ));
        }
        if self.cooldown_ms < 1 || self.cooldown_ms > 86_400_000 {
            return Err(format!(
                "cooldown_ms must be in [1, 86400000], got {}",
                self.cooldown_ms
            ));
        }
        if self.class_enabled.len() != NUM_CLASSES {
            return Err(format!(
                "class_enabled length must be {}, got {}",
                NUM_CLASSES,
                self.class_enabled.len()
            ));
        }
        Ok(())
    }

    /// Deserialize from JSON and validate all fields.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let rules: Self = serde_json::from_str(json).map_err(|e| e.to_string())?;
        rules.validate()?;
        Ok(rules)
    }

    /// Human-readable summary of the current alert configuration.
    pub fn describe(&self) -> String {
        let enabled: Vec<&str> = self
            .class_enabled
            .iter()
            .enumerate()
            .filter(|(_, &on)| on)
            .map(|(i, _)| CLASS_NAMES[i])
            .collect();

        let classes = if enabled.is_empty() {
            "none".to_string()
        } else {
            enabled.join(", ")
        };

        let pct = (self.min_confidence * 100.0) as u32;
        let cooldown_s = self.cooldown_ms / 1000;

        format!(
            "Alerting on: {} (min confidence: {}%, cooldown: {}s)",
            classes, pct, cooldown_s
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::model::{Classification, NUM_CLASSES};
    use crate::vision::motion::MotionBox;

    fn make_detection(class_id: usize, confidence: f32) -> Detection {
        Detection {
            bbox: MotionBox {
                x: 0,
                y: 0,
                w: 32,
                h: 32,
                pixel_count: 100,
            },
            classification: Classification {
                class_id,
                class_name: CLASS_NAMES[class_id],
                confidence,
                scores: {
                    let mut s = [0.0f32; NUM_CLASSES];
                    s[class_id] = confidence;
                    s
                },
            },
            timestamp_ms: 0,
        }
    }

    #[test]
    fn test_default_rules() {
        let rules = AlertRules::default();
        assert!(!rules.class_enabled[0], "nothing should be off");
        assert!(rules.class_enabled[1], "person should be on");
        assert!(rules.class_enabled[2], "vehicle should be on");
        assert!(!rules.class_enabled[3], "animal should be off");
        assert!((rules.min_confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(rules.cooldown_ms, 30_000);
        assert_eq!(rules.mute_until_ms, 0);
        assert_eq!(rules.last_alert_ms, 0);
    }

    #[test]
    fn test_should_alert_person() {
        let rules = AlertRules::default();
        let det = make_detection(1, 0.9); // person, high confidence
        assert!(rules.should_alert(&det, 1000));
    }

    #[test]
    fn test_should_not_alert_nothing() {
        let rules = AlertRules::default();
        let det = make_detection(0, 0.9); // nothing class
        assert!(!rules.should_alert(&det, 1000));
    }

    #[test]
    fn test_should_not_alert_low_confidence() {
        let rules = AlertRules::default();
        let det = make_detection(1, 0.3); // person, low confidence
        assert!(!rules.should_alert(&det, 1000));
    }

    #[test]
    fn test_mute() {
        let mut rules = AlertRules::default();
        rules.mute_until_ms = 200_000;

        let det = make_detection(1, 0.9);

        // Alert during mute period should be blocked
        assert!(!rules.should_alert(&det, 100_000));

        // Alert after mute period should pass
        assert!(rules.should_alert(&det, 300_000));
    }

    #[test]
    fn test_cooldown() {
        let mut rules = AlertRules::default();
        rules.cooldown_ms = 60_000;
        rules.last_alert_ms = 100_000;

        let det = make_detection(1, 0.9);

        // Within cooldown: blocked
        assert!(!rules.should_alert(&det, 120_000));

        // After cooldown: passes
        assert!(rules.should_alert(&det, 200_000));
    }

    #[test]
    fn test_enable_disable_class() {
        let mut rules = AlertRules::default();

        // Enable animal
        assert!(rules.enable_class("animal"));
        assert!(rules.class_enabled[3]);

        // Disable person
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
        assert!(desc.contains("person"), "should mention person: {desc}");
        assert!(desc.contains("vehicle"), "should mention vehicle: {desc}");
        assert!(desc.contains("50%"), "should mention 50%: {desc}");
    }

    #[test]
    fn test_validate_default_ok() {
        assert!(AlertRules::default().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_confidence() {
        let mut rules = AlertRules::default();
        rules.min_confidence = 0.0;
        assert!(rules.validate().is_err());
    }

    #[test]
    fn test_validate_confidence_above_one() {
        let mut rules = AlertRules::default();
        rules.min_confidence = 1.01;
        assert!(rules.validate().is_err());
    }

    #[test]
    fn test_validate_cooldown_zero() {
        let mut rules = AlertRules::default();
        rules.cooldown_ms = 0;
        assert!(rules.validate().is_err());
    }

    #[test]
    fn test_validate_cooldown_too_large() {
        let mut rules = AlertRules::default();
        rules.cooldown_ms = 86_400_001;
        assert!(rules.validate().is_err());
    }

    #[test]
    fn test_from_json_valid() {
        let json = r#"{"class_enabled":[false,true,true,false],"min_confidence":0.7,"mute_until_ms":0,"cooldown_ms":10000}"#;
        let rules = AlertRules::from_json(json).unwrap();
        assert!((rules.min_confidence - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_json_invalid_confidence() {
        let json = r#"{"class_enabled":[false,true,true,false],"min_confidence":0.0,"mute_until_ms":0,"cooldown_ms":10000}"#;
        assert!(AlertRules::from_json(json).is_err());
    }
}
