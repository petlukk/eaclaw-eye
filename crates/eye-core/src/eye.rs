use crate::vision::capture::Camera;
use crate::vision::dedup::DetectionDedup;
use crate::vision::model::Model;
use crate::vision::pipeline::{Detection, Pipeline};
use crate::vision::rules::AlertRules;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum EyeEvent {
    Alert(Detection),
    Error(String),
    Status(String),
}

pub struct EyeState {
    pub rules: AlertRules,
    pub detection_count: u64,
    pub alert_count: u64,
    pub running: bool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Start an async camera loop that processes frames and emits events.
///
/// Returns a receiver for events and a shared handle to the loop state.
/// Set `state.running = false` to stop the loop gracefully.
pub fn start_camera_loop(
    mut camera: Box<dyn Camera>,
    model: Model,
    frame_interval_ms: u64,
) -> (mpsc::Receiver<EyeEvent>, Arc<Mutex<EyeState>>) {
    let (tx, rx) = mpsc::channel(64);

    let state = Arc::new(Mutex::new(EyeState {
        rules: AlertRules::default(),
        detection_count: 0,
        alert_count: 0,
        running: true,
    }));

    let shared_state = Arc::clone(&state);

    tokio::spawn(async move {
        let (w, h) = camera.resolution();
        let mut pipeline = Pipeline::new(model, w, h);
        let mut dedup = DetectionDedup::new(256);

        let _ = tx.send(EyeEvent::Status("Camera started".to_string())).await;

        loop {
            // Check if we should keep running
            {
                let st = shared_state.lock().unwrap();
                if !st.running {
                    break;
                }
            }

            match pipeline.process(camera.as_mut()) {
                Ok(detections) => {
                    for detection in detections {
                        let now = now_ms();

                        // Always count every detection
                        {
                            let mut st = shared_state.lock().unwrap();
                            st.detection_count += 1;
                        }

                        // Check rules and dedup before alerting
                        let should_alert = {
                            let st = shared_state.lock().unwrap();
                            st.rules.should_alert(&detection, now)
                        };

                        if should_alert && !dedup.is_duplicate(&detection) {
                            {
                                let mut st = shared_state.lock().unwrap();
                                st.rules.record_alert(now);
                                st.alert_count += 1;
                            }
                            if tx.send(EyeEvent::Alert(detection)).await.is_err() {
                                // Receiver dropped, stop the loop
                                let mut st = shared_state.lock().unwrap();
                                st.running = false;
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(EyeEvent::Error(e)).await;
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(frame_interval_ms)).await;
        }
    });

    (rx, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::capture::MockCamera;
    use crate::vision::model::Model;

    #[tokio::test]
    async fn test_camera_loop_starts() {
        let camera = Box::new(MockCamera::new(320, 240));
        let model = Model::dummy();

        let (mut rx, state) = start_camera_loop(camera, model, 10);

        // Should receive the "Camera started" status event
        let event = tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            rx.recv(),
        )
        .await
        .expect("timed out waiting for event")
        .expect("channel closed unexpectedly");

        match event {
            EyeEvent::Status(msg) => assert_eq!(msg, "Camera started"),
            other => panic!("expected Status(\"Camera started\"), got {:?}", other),
        }

        // Stop the loop
        {
            let mut st = state.lock().unwrap();
            st.running = false;
        }
    }

    #[tokio::test]
    async fn test_camera_loop_state() {
        let camera = Box::new(MockCamera::new(320, 240));
        let model = Model::dummy();

        let (mut rx, state) = start_camera_loop(camera, model, 10);

        // Wait for the start event to confirm the loop is running
        let _ = tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            rx.recv(),
        )
        .await
        .expect("timed out waiting for start event");

        // Verify state shows running
        {
            let st = state.lock().unwrap();
            assert!(st.running, "state.running should be true after start");
        }

        // Stop the loop
        {
            let mut st = state.lock().unwrap();
            st.running = false;
        }
    }
}
