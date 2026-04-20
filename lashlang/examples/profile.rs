mod bench_support;

use bench_support::{BenchHost, Scenario, benchmark_program, seeded_state};
use lashlang::{
    ExecutionOutcome, ExecutionScratch, ProfileReport, compile_program, parse,
    profile_compiled_with_scratch,
};
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let first = args.next();
    let second = args.next();

    let (scenario, iterations) = match (first.as_deref(), second.as_deref()) {
        (Some(value), maybe_iterations) if Scenario::parse(value).is_some() => (
            Scenario::parse(value).expect("scenario was checked"),
            maybe_iterations
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000),
        ),
        (Some(iterations), None) => (
            Scenario::Baseline,
            iterations.parse::<usize>().ok().unwrap_or(10_000),
        ),
        _ => (Scenario::Baseline, 10_000),
    };

    let source = benchmark_program(scenario);
    let host = BenchHost;
    let program = parse(source).expect("benchmark program should parse");
    let compiled = compile_program(&program);
    let mut profile = ProfileReport::default();
    let mut scratch = ExecutionScratch::new();

    for _ in 0..iterations {
        let mut state = seeded_state();
        let (outcome, run_profile) =
            profile_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                .expect("profiled execution");
        profile.merge(&run_profile);
        let ExecutionOutcome::Finished(_) = outcome else {
            panic!("benchmark program must finish");
        };
    }

    println!("lashlang profile");
    println!("scenario: {scenario}");
    println!("iterations: {iterations}");
    println!("program_bytes: {}", source.len());
    println!();
    println!("instruction_hotspots:");
    for stat in profile.instruction_stats().iter().take(12) {
        println!(
            "{:<16} count={:<10} total_ms={:<10.3} avg_ns={}",
            stat.name,
            stat.count,
            stat.total_ns as f64 / 1_000_000.0,
            stat.avg_ns()
        );
    }
    println!();
    println!("builtin_hotspots:");
    for stat in profile.builtin_stats().iter().take(12) {
        println!(
            "{:<16} count={:<10} total_ms={:<10.3} avg_ns={}",
            stat.name,
            stat.count,
            stat.total_ns as f64 / 1_000_000.0,
            stat.avg_ns()
        );
    }
}
