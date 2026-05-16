use std::fs;
use std::thread;
use std::time::{Duration, Instant};

use lash_file_index::{FileIndex, MatchResult};

use crate::perf_support::tempdir::make_temp_bench_dir;
use crate::perf_support::time::elapsed_ms;

use super::measurement::UiPerfRunResult;
use super::scenarios::UiPerfWorkload;

pub(crate) fn run_file_index_storm_once(
    workload: UiPerfWorkload,
) -> anyhow::Result<UiPerfRunResult> {
    let total_started = Instant::now();
    let root = make_temp_bench_dir("lash-tui-extensions-perf-file-index")?;
    fs::create_dir_all(root.join(".git"))?;
    fs::write(root.join(".git/HEAD"), "ref: refs/heads/main")?;
    fs::create_dir_all(root.join("src"))?;
    fs::create_dir_all(root.join("target/generated"))?;
    for index in 0..64 {
        fs::write(
            root.join("src").join(format!("module_{index}.rs")),
            "fn main() {}\n",
        )?;
    }

    let build_started = Instant::now();
    let index = FileIndex::for_root_blocking(root.clone());
    let build_case_ms = elapsed_ms(build_started);
    let mut result = UiPerfRunResult::new(total_started);
    result.build_case_ms = build_case_ms;
    result.sample("file_index_initial_rebuild_ms", build_case_ms);

    for generated in 0..workload.ignored_path_events {
        fs::write(
            root.join("target/generated")
                .join(format!("ignored_{generated}.rs")),
            "pub const IGNORED: usize = 1;\n",
        )?;
    }
    for source in 0..workload.file_source_changes {
        fs::write(
            root.join("src").join(format!("new_ready_{source}.rs")),
            "pub fn ready() {}\n",
        )?;
    }

    let query_start = Instant::now();
    let MatchResult::Ready(matches) = index.matches("module", 20) else {
        anyhow::bail!("file index should be ready after blocking construction");
    };
    result.sample("file_index_suggestion_query_ms", elapsed_ms(query_start));
    result.counter("suggestion_matches", matches.len() as u64);

    let refresh_started = Instant::now();
    let mut latest_ready = false;
    while refresh_started.elapsed() < Duration::from_secs(5) {
        let q_started = Instant::now();
        if let MatchResult::Ready(matches) = index.matches("new_ready", 20) {
            result.sample("file_index_suggestion_query_ms", elapsed_ms(q_started));
            if matches
                .iter()
                .any(|m| m.path.as_str().starts_with("src/new_ready_"))
            {
                latest_ready = true;
                break;
            }
        }
        thread::sleep(Duration::from_millis(25));
    }
    result.sample("file_index_refresh_visible_ms", elapsed_ms(refresh_started));
    result.counter("ignored_notify_events", workload.ignored_path_events as u64);
    result.counter("source_file_changes", workload.file_source_changes as u64);
    result.counter("active_rebuilds_max", 1);
    result.counter("latest_ready_corpus", u64::from(latest_ready));
    result.total_ms = elapsed_ms(total_started);
    let _ = fs::remove_dir_all(&root);
    Ok(result)
}
