#[path = "../examples/bench_support/mod.rs"]
mod bench_support;

use bench_support::{BenchHost, Scenario, benchmark_program, seeded_state};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lashlang::{
    CompiledProgramCache, ExecutionOutcome, State, Value, compile_program, execute,
    execute_compiled, parse,
};
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
        if matches!(scenario, Scenario::Baseline | Scenario::LanguageSurface) {
            benchmark_full_block_modes(&mut group, &rt, &host, *scenario);
        } else {
            benchmark_execute_only(&mut group, &rt, &host, *scenario);
        }
    }

    group.finish();
}

fn benchmark_full_block_modes(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &tokio::runtime::Runtime,
    host: &BenchHost,
    scenario: Scenario,
) {
    let source = benchmark_program(scenario);
    let program = parse(source).expect("benchmark program should parse");
    let compiled = compile_program(&program);

    group.bench_function(BenchmarkId::new("parse_execute", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let outcome = rt
                .block_on(execute(black_box(source), &mut state, host))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("cached_block", scenario), |b| {
        let mut cache = CompiledProgramCache::default();
        cache
            .get_or_compile(source)
            .expect("benchmark program should compile");
        b.iter(|| {
            let mut state = seeded_state();
            let compiled = cache
                .get_or_compile(black_box(source))
                .expect("benchmark cache lookup");
            let outcome = rt
                .block_on(execute_compiled(black_box(&compiled), &mut state, host))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("cached_session_block", scenario), |b| {
        let mut cache = CompiledProgramCache::default();
        cache
            .get_or_compile(source)
            .expect("benchmark program should compile");
        let mut state = seeded_state();
        b.iter(|| {
            let compiled = cache
                .get_or_compile(black_box(source))
                .expect("benchmark cache lookup");
            let outcome = rt
                .block_on(execute_compiled(black_box(&compiled), &mut state, host))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("execute_only", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let outcome = rt
                .block_on(execute_compiled(black_box(&compiled), &mut state, host))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });

    group.bench_function(BenchmarkId::new("snapshot_execute", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let snapshot = state.snapshot();
            let encoded = serde_json::to_vec(&snapshot).expect("snapshot encode");
            let decoded = serde_json::from_slice(&encoded).expect("snapshot decode");
            state = State::from_snapshot(decoded);
            let outcome = rt
                .block_on(execute_compiled(black_box(&compiled), &mut state, host))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });
}

fn benchmark_execute_only(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    rt: &tokio::runtime::Runtime,
    host: &BenchHost,
    scenario: Scenario,
) {
    let source = benchmark_program(scenario);
    let program = parse(source).expect("benchmark program should parse");
    let compiled = compile_program(&program);

    group.bench_function(BenchmarkId::new("execute_only", scenario), |b| {
        b.iter(|| {
            let mut state = seeded_state();
            let outcome = rt
                .block_on(execute_compiled(black_box(&compiled), &mut state, host))
                .expect("benchmark execution");
            black_box(expect_finished(outcome));
        });
    });
}

fn expect_finished(outcome: ExecutionOutcome) -> Value {
    match outcome {
        ExecutionOutcome::Finished(value) => value,
        ExecutionOutcome::Continued => panic!("benchmark program must finish"),
    }
}

criterion_group!(benches, lashlang_benchmarks);
criterion_main!(benches);
