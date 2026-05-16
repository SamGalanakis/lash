use std::path::{Path, PathBuf};

use super::paths::default_dhat_output_path;
use super::report::ensure_parent_dir;

pub(crate) fn resolve_dhat_output_path(
    enable_dhat: bool,
    report_out: &Path,
    dhat_out: Option<PathBuf>,
    fallback_stem: &str,
) -> Option<PathBuf> {
    if enable_dhat {
        Some(dhat_out.unwrap_or_else(|| default_dhat_output_path(report_out, fallback_stem)))
    } else {
        None
    }
}

pub(crate) fn ensure_dhat_parent(path: Option<&PathBuf>) -> anyhow::Result<()> {
    if let Some(path) = path {
        ensure_parent_dir(path, "dhat output")?;
    }
    Ok(())
}

#[cfg(feature = "dhat-heap")]
pub(crate) fn start_dhat_profiler(
    dhat_out: Option<PathBuf>,
    dhat_frames: Option<usize>,
    _feature_error: &'static str,
) -> anyhow::Result<Option<dhat::Profiler>> {
    let Some(path) = dhat_out else {
        return Ok(None);
    };
    let profiler = dhat::Profiler::builder()
        .file_name(path)
        .trim_backtraces(dhat_frames)
        .build();
    Ok(Some(profiler))
}

#[cfg(not(feature = "dhat-heap"))]
pub(crate) fn start_dhat_profiler(
    dhat_out: Option<PathBuf>,
    _dhat_frames: Option<usize>,
    feature_error: &'static str,
) -> anyhow::Result<Option<()>> {
    if dhat_out.is_some() {
        anyhow::bail!(feature_error);
    }
    Ok(None)
}

#[cfg(feature = "dhat-heap")]
pub(crate) fn finish_dhat_profiler(profiler: Option<dhat::Profiler>) {
    drop(profiler);
}

#[cfg(not(feature = "dhat-heap"))]
pub(crate) fn finish_dhat_profiler(_profiler: Option<()>) {}
