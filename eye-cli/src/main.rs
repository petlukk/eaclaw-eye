use eaclaw_core::agent::Agent;
use eaclaw_core::config::Config;
use eaclaw_core::llm::anthropic::AnthropicProvider;
use eaclaw_core::safety::SafetyLayer;
use eaclaw_core::tools::ToolRegistry;
use eye_core::eye::{start_camera_loop, EyeEvent};
use eye_core::vision::capture::MockCamera;
use eye_core::vision::model::Model;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("eaclaw-eye {}", env!("CARGO_PKG_VERSION"));
        std::process::exit(0);
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "eaclaw-eye {} — Cache-Resident Security Camera",
            env!("CARGO_PKG_VERSION")
        );
        println!();
        println!("Usage: eaclaw-eye [OPTIONS]");
        println!();
        println!("Options:");
        println!("  --whatsapp    Run in WhatsApp mode (scan QR in terminal)");
        println!("  --mock        Use mock camera (for testing without hardware)");
        println!("  --version     Print version");
        println!("  --help        Print this help");
        std::process::exit(0);
    }

    if let Err(e) = eaclaw_core::kernels::init() {
        eprintln!("Failed to initialize SIMD kernels: {e}");
        std::process::exit(1);
    }

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Configuration error: {e}");
            eprintln!("Set ANTHROPIC_API_KEY to get started.");
            std::process::exit(1);
        }
    };

    let use_mock = args.iter().any(|a| a == "--mock");

    let model = Model::dummy();
    eprintln!(
        "Model loaded: {} bytes (fits L1: {})",
        model.size_bytes(),
        if model.size_bytes() < 65536 {
            "yes"
        } else {
            "no"
        }
    );

    let camera: Box<dyn eye_core::vision::capture::Camera> = if use_mock {
        eprintln!("Using mock camera (640x480)");
        Box::new(MockCamera::new(640, 480))
    } else {
        #[cfg(feature = "v4l2")]
        {
            let dev =
                std::env::var("EACLAW_CAMERA").unwrap_or_else(|_| "/dev/video0".into());
            let cam = eye_core::vision::v4l2::V4l2Camera::open(&dev, 640, 480)
                .unwrap_or_else(|e| {
                    eprintln!("Camera error: {e}");
                    std::process::exit(1);
                });
            eprintln!("V4L2 camera: {dev}");
            Box::new(cam)
        }
        #[cfg(not(feature = "v4l2"))]
        {
            eprintln!("Built without v4l2 feature. Use --mock or rebuild with: cargo build --features v4l2");
            std::process::exit(1);
        }
    };

    let (cam_w, cam_h) = camera.resolution();
    eprintln!("Camera: {}x{}", cam_w, cam_h);

    let (mut eye_rx, eye_state) = start_camera_loop(camera, model, 100);

    let llm: Arc<dyn eaclaw_core::llm::LlmProvider> =
        Arc::new(AnthropicProvider::new(&config));
    let tools = ToolRegistry::with_defaults(&config, llm.clone());
    let safety = SafetyLayer::new();

    eprintln!("eaclaw-eye ready. Camera loop running.");

    let channel = eaclaw_core::channel::repl::ReplChannel::new("eaclaw-eye");
    let mut agent = Agent::new(config.clone(), llm, tools, safety);

    tokio::spawn(async move {
        while let Some(event) = eye_rx.recv().await {
            match event {
                EyeEvent::Alert(d) => {
                    eprintln!("[ALERT] {}", d.alert_text());
                }
                EyeEvent::Error(e) => {
                    eprintln!("[ERROR] Camera: {e}");
                }
                EyeEvent::Status(s) => {
                    eprintln!("[STATUS] {s}");
                }
            }
        }
    });

    let result = agent.run(&channel).await;
    channel.shutdown();
    eye_state.lock().unwrap().running = false;

    if let Err(e) = result {
        eprintln!("Agent error: {e}");
        std::process::exit(1);
    }
}
