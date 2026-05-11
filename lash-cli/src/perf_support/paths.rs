use std::path::{Path, PathBuf};

use chrono::Utc;

pub(crate) fn default_report_path(kind: &str) -> PathBuf {
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    repo_root()
        .join(".benchmarks")
        .join(kind)
        .join(format!("{stamp}.json"))
}

pub(crate) fn default_dhat_output_path(report_out: &Path, fallback_stem: &str) -> PathBuf {
    let stem = report_out
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(fallback_stem);
    report_out.with_file_name(format!("{stem}.dhat.json"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("lash-cli crate should live under repo root")
        .to_path_buf()
}
