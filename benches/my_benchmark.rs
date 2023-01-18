use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lap_jv::LapJV;
use rand::Rng;
pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let sizes = vec![10, 20, 50, 100, 1_000];
    for size in sizes {
        c.bench_function(&format!("Problem Size: {size}"), |b| {
            b.iter(|| {
                let mut matrix = Vec::with_capacity(size);
                for _ in 0..size {
                    let mut row = Vec::with_capacity(size);
                    for _ in 0..size {
                        row.push(rng.gen::<f64>())
                    }
                    matrix.push(row);
                }
                LapJV::new(&matrix).solve();
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
