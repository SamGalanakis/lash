use std::path::PathBuf;

use anyhow::Context;
use chrono::Utc;

pub(crate) fn make_temp_bench_dir(prefix: &str) -> anyhow::Result<PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "{}-{}-{}",
        prefix,
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
    Ok(root)
}
