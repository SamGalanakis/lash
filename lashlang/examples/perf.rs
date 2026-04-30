mod bench_support;

use bench_support::{BenchHost, Scenario, benchmark_program, seeded_state};
use lashlang::{
    CompiledProgramCache, ExecutionOutcome, ExecutionScratch, State, compile_program,
    compile_source, execute_compiled_with_scratch, parse, prewarm,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size() as u64);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        record_dealloc(layout.size() as u64);
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, old_layout, new_size) };
        if ptr.is_null() {
            return ptr;
        }
        let old_size = old_layout.size() as u64;
        let new_size = new_size as u64;
        if new_size > old_size {
            record_alloc(new_size - old_size);
        } else {
            record_dealloc(old_size - new_size);
        }
        ptr
    }
}

fn record_alloc(bytes: u64) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            live,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => peak = next,
        }
    }
}

fn record_dealloc(bytes: u64) {
    let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
        Some(live.saturating_sub(bytes))
    });
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Execute,
    Parse,
    Compile,
    Block,
    CachedBlock,
    CachedSessionBlock,
    ColdOnce,
    PrewarmedOnce,
    CachedColdOnce,
    Snapshot,
}

fn main() {
    let mut args = env::args().skip(1);
    let mode = args
        .next()
        .as_deref()
        .map(parse_mode)
        .unwrap_or(Mode::Execute);
    let scenario = args
        .next()
        .as_deref()
        .map(|value| {
            Scenario::parse(value).unwrap_or_else(|| {
                panic!(
                    "unknown scenario `{value}`; expected {}",
                    Scenario::expected_values()
                )
            })
        })
        .unwrap_or(Scenario::Baseline);
    let iterations = args
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(match mode {
            Mode::Parse | Mode::Compile | Mode::Block => 25_000,
            Mode::ColdOnce | Mode::PrewarmedOnce | Mode::CachedColdOnce => 1,
            Mode::Execute | Mode::CachedBlock | Mode::CachedSessionBlock | Mode::Snapshot => {
                100_000
            }
        });

    let source = benchmark_program(scenario);
    let host = BenchHost;
    let mut scratch = ExecutionScratch::new();
    let mut cache = CompiledProgramCache::default();

    reset_alloc_counters();
    let mut started = Instant::now();
    match mode {
        Mode::Execute => {
            let program = parse(source).expect("benchmark program should parse");
            let compiled = compile_program(&program);
            for _ in 0..iterations {
                let mut state = seeded_state();
                let outcome =
                    execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                        .expect("benchmark execution should succeed");
                expect_finished(outcome);
            }
        }
        Mode::Parse => {
            for _ in 0..iterations {
                let program = parse(std::hint::black_box(source)).expect("parse should succeed");
                std::hint::black_box(program);
            }
        }
        Mode::Compile => {
            for _ in 0..iterations {
                let compiled =
                    compile_source(std::hint::black_box(source)).expect("compile should succeed");
                std::hint::black_box(compiled);
            }
        }
        Mode::Block => {
            for _ in 0..iterations {
                let mut state = seeded_state();
                let compiled =
                    compile_source(std::hint::black_box(source)).expect("compile should succeed");
                let outcome =
                    execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                        .expect("benchmark execution should succeed");
                expect_finished(outcome);
            }
        }
        Mode::CachedBlock => {
            cache
                .get_or_compile(source)
                .expect("benchmark program should compile");
            reset_alloc_counters();
            started = Instant::now();
            for _ in 0..iterations {
                let mut state = seeded_state();
                let compiled = cache
                    .get_or_compile(std::hint::black_box(source))
                    .expect("cached compile should succeed");
                let outcome =
                    execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                        .expect("benchmark execution should succeed");
                expect_finished(outcome);
            }
        }
        Mode::CachedSessionBlock => {
            cache
                .get_or_compile(source)
                .expect("benchmark program should compile");
            let mut state = seeded_state();
            reset_alloc_counters();
            started = Instant::now();
            for _ in 0..iterations {
                let compiled = cache
                    .get_or_compile(std::hint::black_box(source))
                    .expect("cached compile should succeed");
                let outcome =
                    execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                        .expect("benchmark execution should succeed");
                expect_finished(outcome);
            }
        }
        Mode::ColdOnce => {
            let mut state = seeded_state();
            let compiled =
                compile_source(std::hint::black_box(source)).expect("compile should succeed");
            let outcome = execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                .expect("benchmark execution should succeed");
            expect_finished(outcome);
        }
        Mode::PrewarmedOnce => {
            prewarm();
            reset_alloc_counters();
            started = Instant::now();
            let mut state = seeded_state();
            let compiled =
                compile_source(std::hint::black_box(source)).expect("compile should succeed");
            let outcome = execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                .expect("benchmark execution should succeed");
            expect_finished(outcome);
        }
        Mode::CachedColdOnce => {
            let mut state = seeded_state();
            let compiled = cache
                .get_or_compile(std::hint::black_box(source))
                .expect("cached compile should succeed");
            let outcome = execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                .expect("benchmark execution should succeed");
            expect_finished(outcome);
        }
        Mode::Snapshot => {
            let program = parse(source).expect("benchmark program should parse");
            let compiled = compile_program(&program);
            for _ in 0..iterations {
                let mut state = seeded_state();
                let snapshot = state.snapshot();
                let encoded = serde_json::to_vec(&snapshot).expect("snapshot encode");
                let decoded = serde_json::from_slice(&encoded).expect("snapshot decode");
                state = State::from_snapshot(decoded);
                let outcome =
                    execute_compiled_with_scratch(&compiled, &mut state, &host, &mut scratch)
                        .expect("benchmark execution should succeed");
                expect_finished(outcome);
            }
        }
    }
    let elapsed = started.elapsed();
    let allocs = alloc_snapshot();

    println!("lashlang perf");
    println!("mode: {mode:?}");
    println!("scenario: {scenario:?}");
    println!("iterations: {iterations}");
    println!("program_bytes: {}", source.len());
    let cache_stats = cache.stats();
    if cache_stats.hits > 0 || cache_stats.misses > 0 {
        println!("cache_hits: {}", cache_stats.hits);
        println!("cache_misses: {}", cache_stats.misses);
        println!("cache_evictions: {}", cache_stats.evictions);
        println!("cache_entries: {}", cache_stats.entries);
    }
    println!("elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!(
        "ns_per_iter: {:.1}",
        elapsed.as_nanos() as f64 / iterations as f64
    );
    println!("allocations: {}", allocs.allocations);
    println!("deallocations: {}", allocs.deallocations);
    println!("allocated_bytes: {}", allocs.allocated_bytes);
    println!(
        "allocations_per_iter: {:.3}",
        allocs.allocations as f64 / iterations as f64
    );
    println!(
        "allocated_bytes_per_iter: {:.1}",
        allocs.allocated_bytes as f64 / iterations as f64
    );
    println!("peak_live_bytes: {}", allocs.peak_live_bytes);
}

fn parse_mode(value: &str) -> Mode {
    match value {
        "execute" => Mode::Execute,
        "parse" => Mode::Parse,
        "compile" => Mode::Compile,
        "block" => Mode::Block,
        "cached_block" => Mode::CachedBlock,
        "cached_session_block" => Mode::CachedSessionBlock,
        "cold_once" => Mode::ColdOnce,
        "prewarmed_once" => Mode::PrewarmedOnce,
        "cached_cold_once" => Mode::CachedColdOnce,
        "snapshot" => Mode::Snapshot,
        other => panic!(
            "unknown mode `{other}`; expected execute, parse, compile, block, cached_block, cached_session_block, cold_once, prewarmed_once, cached_cold_once, or snapshot"
        ),
    }
}

fn reset_alloc_counters() {
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
}

fn alloc_snapshot() -> AllocSnapshot {
    AllocSnapshot {
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
    }
}

struct AllocSnapshot {
    allocated_bytes: u64,
    peak_live_bytes: u64,
    allocations: u64,
    deallocations: u64,
}

fn expect_finished(outcome: ExecutionOutcome) {
    let ExecutionOutcome::Finished(value) = outcome else {
        panic!("benchmark program must finish");
    };
    std::hint::black_box(value);
}
