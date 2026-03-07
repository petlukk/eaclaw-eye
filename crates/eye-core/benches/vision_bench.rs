use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use eye_core::vision::frame::Frame;
use eye_core::vision::inference::InferenceEngine;
use eye_core::vision::model::Model;
use eye_core::vision::motion::MotionDetector;

fn bench_frame_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("motion_detection");
    let mut det = MotionDetector::new(25, 10);

    for &(w, h) in &[(640, 480), (1280, 720), (1920, 1080)] {
        let f1 = Frame::new(w, h);
        let mut f2 = Frame::new(w, h);
        for (i, p) in f2.data.iter_mut().enumerate() {
            *p = ((i * 7) % 256) as u8;
        }

        group.throughput(Throughput::Bytes((w * h) as u64));
        group.bench_function(
            format!("{w}x{h}"),
            |b| b.iter(|| det.detect(black_box(&f1), black_box(&f2))),
        );
    }
    group.finish();
}

fn bench_crop_and_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("crop_and_scale");
    let mut f = Frame::new(640, 480);
    for (i, p) in f.data.iter_mut().enumerate() {
        *p = (i % 256) as u8;
    }

    group.bench_function("640x480_to_64x64", |b| {
        b.iter(|| f.crop_and_scale(black_box(100), black_box(100), 200, 200, 64))
    });
    group.finish();
}

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnn_inference");
    let mut engine = InferenceEngine::new();
    let model = Model::dummy();
    let mut frame = Frame::new(64, 64);
    for (i, p) in frame.data.iter_mut().enumerate() {
        *p = (i % 256) as u8;
    }

    group.bench_function("64x64_classify", |b| {
        b.iter(|| engine.classify(black_box(&frame), black_box(&model)))
    });
    group.finish();
}

criterion_group!(benches, bench_frame_diff, bench_crop_and_scale, bench_inference);
criterion_main!(benches);
