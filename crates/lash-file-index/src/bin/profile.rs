//! Microbenchmark for `FileIndex` hot paths.
//!
//! Usage:
//!   `cargo run -p lash-file-index --release --bin profile -- [root]`
//!   `cargo run -p lash-file-index --release --bin profile -- --synth 50000`
//!
//! Walks the given root (default: cwd), then runs N iterations each of:
//!   * empty-query matches (sort-the-whole-corpus path)
//!   * a few non-empty fuzzy queries (matcher + scoring + top-K path)
//!
//! With `--synth N`, materializes N fake files under a tempdir to stress-test
//! the per-keystroke cost on a large corpus without needing a real giant repo.
//!
//! Prints walk time, corpus size, and per-call timings (mean / p50 / p95).

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use lash_file_index::{FileIndex, MatchResult};

/// Tempdir helper local to this bin so the crate doesn't need `tempfile` as
/// a runtime dep. Directory is removed on drop.
struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let path = env::temp_dir().join(format!(
            "lash-file-index-profile-{}-{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos(),
        ));
        fs::create_dir_all(&path).expect("create tempdir");
        Self(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const ITERATIONS: usize = 200;
const QUERIES: &[&str] = &[
    "input_handling",
    "editor",
    "shell",
    "tui",
    "ts",
    "rs",
    "cargo",
    "test",
];

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let (root, _hold_tempdir) = match args.as_slice() {
        [flag, n] if flag == "--synth" => {
            let n: usize = n.parse().expect("--synth expects an integer");
            let dir = synth_corpus(n);
            (dir.path().to_path_buf(), Some(dir))
        }
        [path] => (PathBuf::from(path), None),
        [] => (env::current_dir().expect("cwd"), None),
        _ => panic!("usage: profile [root | --synth N]"),
    };
    println!("root: {}", root.display());

    let walk_start = Instant::now();
    let index = FileIndex::for_root_blocking(root);
    let walk_elapsed = walk_start.elapsed();

    // Force a Ready read to learn corpus size via empty query.
    let MatchResult::Ready(initial) = index.matches("", 1) else {
        panic!("expected Ready after for_root_blocking");
    };
    let _ = initial;

    // The blocking constructor returns once on_ready fires; the walker thread
    // then morphs into the watcher loop. Probe the corpus size by doing an
    // empty match with a huge limit (one-shot, off the hot-path measurement).
    let corpus_size = match index.matches("", usize::MAX) {
        MatchResult::Ready(v) => v.len(),
        _ => 0,
    };
    println!("walk: {:?}  corpus: {} entries", walk_elapsed, corpus_size);

    println!("\nEmpty query (`@` with nothing typed):");
    bench(&index, "", ITERATIONS);

    println!("\nNon-empty queries (warm matcher):");
    for q in QUERIES {
        bench(&index, q, ITERATIONS);
    }
}

/// Materialize a synthetic corpus of `n` files distributed across a directory
/// tree. Filename diversity matters for fuzzy-match realism, so we sprinkle
/// language-flavored substrings.
fn synth_corpus(n: usize) -> TempDir {
    let dir = TempDir::new();
    let modules = [
        "editor",
        "renderer",
        "parser",
        "lexer",
        "compiler",
        "runtime",
        "input",
        "output",
        "session",
        "transport",
        "store",
        "cache",
        "queue",
        "pool",
        "config",
        "logger",
        "tracer",
        "metrics",
        "api",
        "service",
        "client",
        "server",
        "router",
        "handler",
        "controller",
        "model",
        "view",
        "component",
        "widget",
        "layout",
    ];
    let exts = ["rs", "ts", "tsx", "py", "go", "md"];
    fs::create_dir_all(dir.path()).unwrap();
    let chunk = 256;
    for i in 0..n {
        let m1 = modules[i % modules.len()];
        let m2 = modules[(i / modules.len()) % modules.len()];
        let ext = exts[i % exts.len()];
        let bucket = i / chunk;
        let parent = dir.path().join(format!("crate{}/src/{}", bucket, m1));
        fs::create_dir_all(&parent).unwrap();
        fs::write(parent.join(format!("{m2}_{i}.{ext}")), "").unwrap();
    }
    dir
}

fn bench(index: &FileIndex, query: &str, iters: usize) {
    let mut samples = Vec::with_capacity(iters);
    let mut hits = 0usize;
    for _ in 0..iters {
        let t = Instant::now();
        let result = index.matches(query, 20);
        let elapsed = t.elapsed();
        if let MatchResult::Ready(v) = result {
            hits = v.len();
        }
        samples.push(elapsed);
    }
    samples.sort_unstable();
    let mean: u128 = samples.iter().map(|d| d.as_nanos()).sum::<u128>() / iters as u128;
    let p50 = samples[iters / 2];
    let p95 = samples[(iters * 95) / 100];
    println!(
        "  query={:<18} hits={:<3}  mean={:>8}µs  p50={:>8}µs  p95={:>8}µs",
        format!("{:?}", query),
        hits,
        mean / 1000,
        p50.as_micros(),
        p95.as_micros(),
    );
}
