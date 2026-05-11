use std::path::Path;

use anyhow::Context;
use serde::Serialize;

pub(crate) fn ensure_parent_dir(path: &Path, label: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} dir {}", parent.display()))?;
    }
    Ok(())
}

pub(crate) fn write_json_report(path: &Path, report: &impl Serialize) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write benchmark report {}", path.display()))
}
