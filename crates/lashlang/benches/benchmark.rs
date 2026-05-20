#[path = "../examples/bench_support/mod.rs"]
mod bench_support;

use bench_support::{BenchHost, Scenario, benchmark_program, projected_bindings, seeded_state_for};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lashlang::{ExecutionEnvironment, ExecutionOutcome, State, Value, compile, execute, prewarm};
use std::hint::black_box;
use std::time::Duration;

fn lashlang_benchmarks(c: &mut Criterion) {
    let host = BenchHost;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut group = c.benchmark_group("lashlang");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(60);

    for scenario in Scenario::ALL {
        benchmark_one_shot_modes(&mut group, &rt, &host, *scenario);
    }

    group.finish();
}

fn benchmark_one_shot_modes(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &tokio::runtime::Runtime,
    host: &BenchHost,
    scenario: Scenario,
) {
    let source = benchmark_program(scenario);
    let compiled = compile(source).expect("benchmark program should compile");
    let projected = projected_bindings(scenario);

    group.bench_function(BenchmarkId::new("one_shot", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state_for(scenario);
            let compiled = compile(black_box(source)).expect("benchmark program should compile");
            let env = ExecutionEnvironment::new(host).with_projected_bindings(projected.clone());
            let outcome = rt
                .block_on(execute(&compiled, &mut state, &env))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("prewarmed_one_shot", scenario), |b| {
        prewarm();
        b.iter(|| {
            let mut state = seeded_state_for(scenario);
            let compiled = compile(black_box(source)).expect("benchmark program should compile");
            let env = ExecutionEnvironment::new(host).with_projected_bindings(projected.clone());
            let outcome = rt
                .block_on(execute(&compiled, &mut state, &env))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("compiled_execute", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state_for(scenario);
            let env = ExecutionEnvironment::new(host).with_projected_bindings(projected.clone());
            let outcome = rt
                .block_on(execute(black_box(&compiled), &mut state, &env))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("snapshot", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state_for(scenario);
            let snapshot = state.snapshot();
            let encoded = serde_json::to_vec(&snapshot).expect("snapshot encode");
            let decoded = serde_json::from_slice(&encoded).expect("snapshot decode");
            state = State::from_snapshot(decoded);
            let env = ExecutionEnvironment::new(host).with_projected_bindings(projected.clone());
            let outcome = rt
                .block_on(execute(black_box(&compiled), &mut state, &env))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });
}

fn expect_finished(outcome: ExecutionOutcome) -> Value {
    match outcome {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => panic!("benchmark program must finish"),
        ExecutionOutcome::Failed(value) => panic!("unexpected process failure: {value}"),
    }
}

criterion_group!(benches, lashlang_benchmarks);
criterion_main!(benches);
