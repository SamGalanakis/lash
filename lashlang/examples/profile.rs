mod bench_support;

use bench_support::{BenchHost, Scenario, benchmark_program, projected_bindings, seeded_state};
use lashlang::{ExecutionOutcome, ExecutionScratch, ProfileReport, compile_program, parse};
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let first = args.next();
    let second = args.next();

    let (scenarios, iterations) = match (first.as_deref(), second.as_deref()) {
        (Some("all"), maybe_iterations) => (
            Scenario::ALL.to_vec(),
            maybe_iterations
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000),
        ),
        (Some(value), maybe_iterations) if Scenario::parse(value).is_some() => (
            vec![Scenario::parse(value).expect("scenario was checked")],
            maybe_iterations
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000),
        ),
        (Some(iterations), None) => (
            vec![Scenario::Baseline],
            iterations.parse::<usize>().ok().unwrap_or(10_000),
        ),
        _ => (vec![Scenario::Baseline], 10_000),
    };

    let host = BenchHost;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let mut profile = ProfileReport::default();
    let mut scratch = ExecutionScratch::new();
    let mut program_bytes = 0usize;

    for scenario in &scenarios {
        let source = benchmark_program(*scenario);
        program_bytes += source.len();
        let program = parse(source).expect("benchmark program should parse");
        let compiled = compile_program(&program);

        for _ in 0..iterations {
            let mut state = seeded_state();
            let (outcome, run_profile) = rt
                .block_on(
                    lashlang::profile_compiled_with_scratch_and_projected_bindings(
                        &compiled,
                        &mut state,
                        &host,
                        &mut scratch,
                        &projected_bindings(*scenario),
                    ),
                )
                .expect("profiled execution");
            profile.merge(&run_profile);
            let ExecutionOutcome::Finished(_) = outcome else {
                panic!("benchmark program must finish");
            };
        }
    }

    println!("lashlang profile");
    if scenarios.len() == 1 {
        println!("scenario: {}", scenarios[0]);
    } else {
        println!(
            "scenario: all ({})",
            scenarios
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("iterations: {iterations}");
    println!("program_bytes: {program_bytes}");
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
