use lash::ToolResultProjectionPluginConfig;
use lash::embedded::{LashlangRequest, LashlangResponse, LashlangRuntime, LashlangToolReply};
use serde_json::Value;
use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, old_layout, new_size) };
        if !ptr.is_null() {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            let old_size = old_layout.size() as u64;
            let new_size = new_size as u64;
            if new_size > old_size {
                ALLOCATED_BYTES.fetch_add(new_size - old_size, Ordering::Relaxed);
            } else {
                DEALLOCATED_BYTES.fetch_add(old_size - new_size, Ordering::Relaxed);
            }
        }
        ptr
    }
}

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

fn main() {
    let iterations = env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000);
    let runtime = LashlangRuntime::start().expect("start embedded lashlang runtime");
    runtime
        .send(LashlangRequest::Init {
            tools_json: "[]".to_string(),
            session_id: "embedded-perf".to_string(),
            observe_projection: ToolResultProjectionPluginConfig::default(),
        })
        .expect("send init");
    expect_ready(&runtime);

    reset_alloc_counters();
    let first_started = Instant::now();
    run_block(&runtime, "first", SOURCE);
    let first_elapsed = first_started.elapsed();
    let first_allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let first_allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);

    reset_alloc_counters();
    let started = Instant::now();
    for _ in 0..iterations {
        run_block(&runtime, "run", SOURCE);
    }
    let elapsed = started.elapsed();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let deallocations = DEALLOCATIONS.load(Ordering::Relaxed);
    let reallocations = REALLOCATIONS.load(Ordering::Relaxed);
    let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let deallocated_bytes = DEALLOCATED_BYTES.load(Ordering::Relaxed);

    println!("embedded lashlang perf");
    println!("iterations: {iterations}");
    println!("program_bytes: {}", SOURCE.len());
    println!(
        "first_exec_ms: {:.3}",
        first_elapsed.as_secs_f64() * 1_000.0
    );
    println!(
        "first_exec_ns: {:.1}",
        first_elapsed.as_secs_f64() * 1_000_000_000.0
    );
    println!("first_exec_allocations: {first_allocations}");
    println!("first_exec_allocated_bytes: {first_allocated_bytes}");
    println!("elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!(
        "ns_per_iter: {:.1}",
        elapsed.as_secs_f64() * 1_000_000_000.0 / iterations as f64
    );
    println!("allocations: {allocations}");
    println!("deallocations: {deallocations}");
    println!("reallocations: {reallocations}");
    println!("allocated_bytes: {allocated_bytes}");
    println!("deallocated_bytes: {deallocated_bytes}");
    println!(
        "allocations_per_iter: {:.3}",
        allocations as f64 / iterations as f64
    );
    println!(
        "allocated_bytes_per_iter: {:.1}",
        allocated_bytes as f64 / iterations as f64
    );
}

fn expect_ready(runtime: &LashlangRuntime) {
    match runtime.recv().expect("receive ready") {
        LashlangResponse::Ready => {}
        other => panic!("expected ready, got {other:?}"),
    }
}

fn run_block(runtime: &LashlangRuntime, id: &str, code: &str) {
    runtime
        .send(LashlangRequest::Exec {
            id: id.to_string(),
            code: code.to_string(),
            accept_finish: true,
        })
        .expect("send exec");

    loop {
        match runtime.recv().expect("receive runtime response") {
            LashlangResponse::ToolCall {
                name,
                args,
                result_tx,
                ..
            } => {
                let result = execute_tool(&name, args);
                result_tx
                    .send(LashlangToolReply {
                        success: result.is_ok(),
                        result: result.unwrap_or_else(Value::String),
                    })
                    .expect("send tool reply");
            }
            LashlangResponse::ToolBatchCall { calls, result_tx } => {
                let replies = calls
                    .into_iter()
                    .map(|call| {
                        let result = execute_tool(&call.name, call.args);
                        LashlangToolReply {
                            success: result.is_ok(),
                            result: result.unwrap_or_else(Value::String),
                        }
                    })
                    .collect();
                result_tx.send(replies).expect("send tool batch reply");
            }
            LashlangResponse::ExecResult {
                id: result_id,
                error,
                terminal_finish,
                ..
            } => {
                assert_eq!(result_id, id);
                assert!(error.is_none(), "execution failed: {error:?}");
                assert!(terminal_finish.is_some(), "expected terminal finish");
                return;
            }
            other => panic!("unexpected runtime response: {other:?}"),
        }
    }
}

fn execute_tool(name: &str, args: Value) -> Result<Value, String> {
    match name {
        "echo" => Ok(args.get("value").cloned().unwrap_or(Value::Null)),
        other => Err(format!("unknown tool: {other}")),
    }
}

fn reset_alloc_counters() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    DEALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

const SOURCE: &str = r#"
items = [
  { label: "alpha", weight: 1, active: true },
  { label: "beta", weight: 2, active: false },
  { label: "gamma", weight: 3, active: true }
]
indexes = range(0, len(items))
all_indexes = push(indexes, len(items))
total = 0
labels = []
for item in items {
  total = total + item.weight
  if item.active {
    labels = labels + [format("{0}:{1}", item.label, item.weight)]
  }
}
fanout = parallel {
  lookup: call echo { value: join(labels, ",") }
  stats: call echo { value: { total: total, count: len(items), seen: 3, index_count: len(all_indexes) } }
}
lookup_value = fanout.lookup?
stats_value = validate(fanout.stats?, Type { total: int, count: int, seen: int, index_count: int })
summary = format(
  "user={0};attempt={1};active={2};total={3};count={4};seen={5};indexes={6}",
  "sam",
  3,
  lookup_value,
  stats_value.total,
  stats_value.count,
  stats_value.seen,
  stats_value.index_count
)
submit summary
"#;
