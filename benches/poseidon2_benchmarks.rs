use std::time::Duration;

use air::AirSettings;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};
use whirlaway::examples::poseidon2::prove_poseidon2_prover_only;

const SAMPLE_SIZE: usize = 10;
const MEASUREMENT_TIME_SECS: u64 = 20;

fn poseidon2_settings(log_inv_rate: usize, univariate_skips: usize) -> AirSettings {
    AirSettings::new(
        128,
        SecurityAssumption::CapacityBound,
        FoldingFactor::ConstantFromSecondRound(7, 4),
        log_inv_rate,
        univariate_skips,
        5,
    )
}

fn bench_poseidon2_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2 prover");
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASUREMENT_TIME_SECS));

    for (log_n_rows, log_inv_rate, univariate_skips) in [(16usize, 1usize, 4usize), (18, 1, 4)] {
        let settings = poseidon2_settings(log_inv_rate, univariate_skips);
        let bench_id = BenchmarkId::new(
            "prove_only",
            format!("rows_2^{log_n_rows}_skips_{univariate_skips}"),
        );

        group.bench_function(bench_id, |b| {
            b.iter_custom(|iters| {
                let mut prover_total = Duration::ZERO;
                for _ in 0..iters {
                    prover_total +=
                        prove_poseidon2_prover_only(log_n_rows, settings.clone(), 0, false)
                            .prover_time;
                }
                prover_total
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_poseidon2_prover);
criterion_main!(benches);
